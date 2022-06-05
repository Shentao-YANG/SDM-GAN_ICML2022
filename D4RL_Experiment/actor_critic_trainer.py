import numpy as np
import torch
import torch.nn.functional as F
import copy
import utils.model_rollout_functions as mrf
from utils.logger import create_stats_ordered_dict, logger
from utils.utils import print_banner
import torch.autograd as autograd


class GanACTrainer(object):

    def __init__(
            self,
            device,
            discount,                       # discount factor
            beta,                           # target network update rate
            actor_lr,                       # actor learning rate
            critic_lr,                      # critic learning rate
            dis_lr,                         # discriminator learning rate
            lmbda,                          # weight of the minimum in Q-update
            log_lagrange,                   # value of log lagrange multiplier
            policy_freq,                    # update frequency of the actor
            state_noise_std,
            num_action_bellman,
            actor,                          # Actor object
            critic,                         # Critic object
            discriminator,                  # Discriminator object
            dynamics_model,                 # Model object, Note that GanACTrainer is not responsible for training this
            replay_buffer,                  # The true replay buffer,
            generated_data_buffer,          # Replay buffer solely consisting of synthetic transitions
            rollout_len_func,               # Rollout length as a function of number of train calls
            rollout_len_fix=1,              # fixed constant rollout length
            num_model_rollouts=512,         # Number of *transitions* to generate per training timestep
            rollout_generation_freq=1,      # Can save time by only generating data when model is updated
            rollout_batch_size=int(1024),   # Maximum batch size for generating rollouts (i.e. GPU memory limit)
            real_data_pct=0.05,             # Percentage of real data used for actor-critic training
            terminal_cutoff=1e8,            # output Done if model pred > terminal_cutoff
            batch_size=256,
            warm_start_epochs=40
    ):
        super().__init__()

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.4, 0.999))

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.discriminator = discriminator
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr, betas=(0.4, 0.999))
        self.adversarial_loss = torch.nn.BCELoss()

        self.log_lagrange = torch.tensor(log_lagrange, device=device)

        self.dynamics_model = dynamics_model
        self.replay_buffer = replay_buffer
        self.generated_data_buffer = generated_data_buffer
        self.rollout_len_func = rollout_len_func
        self.rollout_len_fix = rollout_len_fix
        self.terminal_cutoff = terminal_cutoff

        self.num_model_rollouts = num_model_rollouts
        self.rollout_generation_freq = rollout_generation_freq
        self.rollout_batch_size = rollout_batch_size
        self.real_data_pct = real_data_pct

        self.device = device
        self.discount = discount
        self.beta = beta
        self.lmbda = lmbda
        self.policy_freq = int(policy_freq)
        self.state_noise_std = state_noise_std
        self.num_action_bellman = num_action_bellman
        self.batch_size = batch_size
        self.warm_start_epochs = warm_start_epochs

        self._n_train_steps_total = 0
        self._n_epochs = 0

        self.Q_average = None
        self.epoch_critic_loss_thres = 1000.

        reward_stat = self.replay_buffer.reward_stat
        self.reward_max = reward_stat['max'] + 3. * reward_stat['std']
        self.reward_min = reward_stat['min'] - 3. * reward_stat['std']

        print_banner(f"reward_max: {self.reward_max:.3f}, reward_min: {self.reward_min:.3f}")
        print_banner("Initialized Model-based GAN Actor-Critic Trainer !")

    def model_based_rollout(self):
        rollout_len = self.rollout_len_func(train_steps=self._n_train_steps_total, n=self.rollout_len_fix)
        total_samples = self.rollout_generation_freq * self.num_model_rollouts

        num_samples, generated_rewards, terminated = 0, np.array([]), []
        while num_samples < total_samples:
            batch_samples = min(self.rollout_batch_size, total_samples - num_samples)
            start_states = self.replay_buffer.random_batch(batch_samples)['observations']       # (batch_samples, state_dim), np.array

            with torch.no_grad():
                paths = mrf.policy(
                    dynamics_model=self.dynamics_model,
                    policy=self.actor,
                    start_states=start_states,
                    device=self.device,
                    max_path_length=rollout_len,
                    terminal_cutoff=self.terminal_cutoff
                )

            for path in paths:
                self.generated_data_buffer.add_path(path)
                num_samples += len(path['observations'])
                generated_rewards = np.concatenate([generated_rewards, path['rewards'][:, 0]])
                terminated.append(path['terminals'][-1, 0] > self.terminal_cutoff)

            if num_samples >= total_samples:
                break

        return generated_rewards, terminated, rollout_len

    def sample_batch(self, n_real_data, n_generated_data, to_gpu=True):
        batch = self.replay_buffer.random_batch(n_real_data)
        generated_batch = self.generated_data_buffer.random_batch(n_generated_data)

        for k in ('rewards', 'terminals', 'observations', 'actions', 'next_observations'):
            batch[k] = np.concatenate((batch[k], generated_batch[k]), axis=0)
            if to_gpu:
                batch[k] = torch.from_numpy(batch[k]).float().to(self.device)

        return batch

    def train_from_torch(self, iterations):
        self._n_epochs += 1
        real_data_pct_curr = min(max(1. - (self._n_epochs - (self.warm_start_epochs + 1)) // 5 * 0.1, self.real_data_pct), 1.)
        n_real_data = int(real_data_pct_curr * self.batch_size)
        n_generated_data = self.batch_size - n_real_data
        warm_start = self._n_epochs <= self.warm_start_epochs
        epoch_critic_loss = []

        """
        Update policy on both real and generated data
        """

        for _ in range(iterations):
            """
            Generate synthetic data using dynamics model
            """
            if self._n_train_steps_total % self.rollout_generation_freq == 0:
                generated_rewards, terminated, rollout_len = self.model_based_rollout()

            """
            Critic Training
            """
            batch = self.sample_batch(n_real_data=n_real_data, n_generated_data=n_generated_data, to_gpu=True)

            with torch.no_grad():
                # Duplicate state num_action_bellman times to sample num_action_bellman actions for each state
                next_state_repeat, next_actions = self.actor_target.sample_multiple_actions(batch['next_observations'], num_action=self.num_action_bellman, std=self.state_noise_std)
                # Compute value of the sampled actions, [Q11;Q12;Q13;Q21;Q22;Q23]
                target_Q = self.critic_target.weighted_min(next_state_repeat, next_actions, lmbda=self.lmbda)
                # find the mean Q-value over the sampled actions for each state
                target_Q = target_Q.view(self.batch_size, -1).mean(1).view(-1, 1)  # (batch_size, 1)
                target_Q = batch['rewards'].clamp(min=self.reward_min, max=self.reward_max) + (1. - batch['terminals']) * self.discount * target_Q * (target_Q.abs() < 2000.)

            current_Q1, current_Q2 = self.critic(batch['observations'], batch['actions'])
            if current_Q1.size() != target_Q.size():  # check dimensions here
                raise ValueError(f"Shape of current_Q1={current_Q1.size()}, shape of target_Q={target_Q.size()}.")

            critic_loss = F.huber_loss(current_Q1, target_Q, delta=500.) + F.huber_loss(current_Q2, target_Q, delta=500.)

            if not warm_start and critic_loss < self.epoch_critic_loss_thres:                      # do not train the critic during the warm start
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            epoch_critic_loss.append(critic_loss.detach().cpu().numpy().mean())

            """
            Actor and Discriminator Training
            """
            batch_fake_state = self.sample_batch(n_real_data=n_real_data, n_generated_data=n_generated_data, to_gpu=False)                # self.generated_data_buffer.random_batch(self.batch_size)
            batch_fake_state = batch_fake_state['observations'][np.where(batch_fake_state["terminals"] < 0.5)[0]]

            # sample noisy state-actions for each fake-state in the mini-batch
            fake_state_repeat, fake_action_samples, fake_raw_action_samples = \
                self.actor.sample_multiple_actions(torch.from_numpy(batch_fake_state).float().to(self.device),
                                                   num_action=1, std=self.state_noise_std, return_raw_action=True)
            with torch.no_grad():
                fake_transitions = self.dynamics_model.sample(torch.cat([fake_state_repeat, fake_action_samples], dim=-1))      # (batch_size, 2 + state_dim)
            if (fake_transitions != fake_transitions).any():
                fake_transitions[fake_transitions != fake_transitions] = 0
            fake_delta_obs = fake_transitions[:, 2:]
            fake_next_state = fake_state_repeat + fake_delta_obs
            fake_not_dones = (fake_transitions[:, 1] <= self.terminal_cutoff)
            fake_next_state = fake_next_state[fake_not_dones]
            _, fake_next_raw_action = self.actor(fake_next_state, return_raw_action=True)

            fake_samples = torch.cat(
                [
                    torch.cat([fake_state_repeat, fake_raw_action_samples], dim=1),             # (s_t, a_t)
                    torch.cat([fake_next_state, fake_next_raw_action], dim=1)                   # (s_{t+1}, a_{t+1})
                ], dim=0
            )
            batch_truth = self.replay_buffer.random_batch(fake_samples.shape[0])                # self.batch_size * 2
            true_samples = torch.cat(
                [   # (batch_truth['observations'], batch_truth['actions'])
                    torch.from_numpy(batch_truth['observations']).float().to(self.device), self.actor.pre_scaling_action(torch.from_numpy(batch_truth['actions']).float().to(self.device))
                ], dim=1
            )
            generator_loss = self.adversarial_loss(self.discriminator(fake_samples),
                                                   torch.ones(fake_samples.size(0), 1, device=self.device))

            # Measure discriminator's ability to classify real from generated samples
            # label smoothing via soft and noisy labels
            fake_labels = torch.zeros(fake_samples.size(0), 1, device=self.device)
            true_labels = torch.rand(size=(true_samples.size(0), 1), device=self.device) * (1.0 - 0.80) + 0.80      # [0.80, 1.0)

            real_loss = self.adversarial_loss(self.discriminator(true_samples), true_labels)
            fake_loss = self.adversarial_loss(self.discriminator(fake_samples.detach()), fake_labels)
            discriminator_loss = (real_loss + fake_loss) / 2

            ### Compute optimization objective for policy update ###
            Q_values = self.critic.q_min(batch['observations'], self.actor(batch['observations']))
            if self.Q_average is None:
                self.Q_average = Q_values.abs().mean().detach()

            if warm_start:
                policy_loss = generator_loss
            else:
                lagrange = self.log_lagrange / self.Q_average
                policy_loss = -lagrange * Q_values.mean() + generator_loss

            # Update policy (actor): minimize policy loss
            if self._n_train_steps_total % self.policy_freq == 0:
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

            self.Q_average = self.beta * Q_values.abs().mean().detach() + (1. - self.beta) * self.Q_average

            self._n_train_steps_total += 1

        print_banner(f"Training epoch: {str(self._n_epochs)}, perform warm_start training: {warm_start}", separator="*", num_star=90)
        assert len(epoch_critic_loss) == iterations, f"len(epoch_critic_loss)={len(epoch_critic_loss)}, should be {iterations}"
        self.epoch_critic_loss_thres = 0.95 * self.epoch_critic_loss_thres + 0.05 * (np.mean(epoch_critic_loss) + 3. * np.std(epoch_critic_loss))
        # Logging
        logger.record_tabular('Num Real Data', n_real_data)
        logger.record_tabular('Num Generated Data', n_generated_data)
        logger.record_tabular('Num True Samples', true_samples.shape[0])
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy()))
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Epoch Critic Loss', np.array(epoch_critic_loss)))
        logger.record_tabular('Epoch Critic Loss Thres', self.epoch_critic_loss_thres)
        logger.record_tabular('Actor Loss', policy_loss.cpu().data.numpy())
        logger.record_tabular('Generator Loss', generator_loss.cpu().data.numpy())
        logger.record_tabular('Q(s,a_sample)', Q_values.mean().cpu().data.numpy())
        logger.record_tabular('Q Average', self.Q_average.cpu().data.numpy())
        logger.record_tabular('Real Loss', real_loss.cpu().data.numpy())
        logger.record_tabular('Fake Loss', fake_loss.cpu().data.numpy())
        logger.record_tabular('Discriminator Loss', discriminator_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Current_Q1', current_Q1.cpu().data.numpy()))
        logger.record_tabular('Rollout Length', rollout_len)
        logger.record_dict(create_stats_ordered_dict('Model Reward Predictions', generated_rewards))
        logger.record_tabular('Model Rollout Terminations', np.mean(terminated))

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.discriminator.state_dict(), '%s/%s_discriminator.pth' % (directory, filename))
        torch.save(self.discriminator_optimizer.state_dict(), '%s/%s_discriminator_optimizer.pth' % (directory, filename))

        torch.save(self.log_lagrange.cpu(), '%s/%s_log_lagrange.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)

        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.discriminator.load_state_dict(torch.load('%s/%s_discriminator.pth' % (directory, filename)))
        self.discriminator_optimizer.load_state_dict(torch.load('%s/%s_discriminator_optimizer.pth' % (directory, filename)))

        self.log_lagrange = torch.load('%s/%s_log_lagrange.pth' % (directory, filename))

    @property
    def networks(self):
        return self.actor

    @property
    def num_train_steps(self):
        return self._n_train_steps_total

    @property
    def num_epochs(self):
        return self._n_epochs

    def get_snapshot(self):
        snapshot = dict(
            dynamics_model=self.dynamics_model,
            actor=self.actor,
            critic=self.critic,
            discriminator=self.discriminator,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            log_lagrange=self.log_lagrange
        )
        return snapshot


class W1ACTrainer(GanACTrainer):

    def __init__(
            self,
            device,
            discount,                       # discount factor
            beta,                           # target network update rate
            actor_lr,                       # actor learning rate
            critic_lr,                      # critic learning rate
            dis_lr,                         # discriminator learning rate
            lmbda,                          # weight of the minimum in Q-update
            log_lagrange,                   # value of log lagrange multiplier
            policy_freq,                    # update frequency of the actor
            state_noise_std,
            num_action_bellman,
            actor,                          # Actor object
            critic,                         # Critic object
            discriminator,                  # Discriminator object
            dynamics_model,                 # Model object, Note that GanACTrainer is not responsible for training this
            replay_buffer,                  # The true replay buffer,
            generated_data_buffer,          # Replay buffer solely consisting of synthetic transitions
            rollout_len_func,               # Rollout length as a function of number of train calls
            rollout_len_fix=1,              # fixed constant rollout length
            num_model_rollouts=512,         # Number of *transitions* to generate per training timestep
            rollout_generation_freq=1,      # Can save time by only generating data when model is updated
            rollout_batch_size=int(1024),   # Maximum batch size for generating rollouts (i.e. GPU memory limit)
            real_data_pct=0.05,             # Percentage of real data used for actor-critic training
            terminal_cutoff=1e8,            # output Done if model pred > terminal_cutoff
            batch_size=256,
            warm_start_epochs=40,
            lmbda_gp=10.
    ):
        super().__init__(
            device=device,
            discount=discount,
            beta=beta,  # target network update rate
            actor_lr=actor_lr,  # actor learning rate
            critic_lr=critic_lr,  # critic learning rate
            dis_lr=dis_lr,  # discriminator learning rate
            lmbda=lmbda,  # weight of the minimum in Q-update
            log_lagrange=log_lagrange,  # value of log lagrange multiplier
            policy_freq=policy_freq,  # update frequency of the actor
            state_noise_std=state_noise_std,
            num_action_bellman=num_action_bellman,
            actor=actor,  # Actor object
            critic=critic,  # Critic object
            discriminator=discriminator,  # Discriminator object
            dynamics_model=dynamics_model,  # Model object, Note that GanACTrainer is not responsible for training this
            replay_buffer=replay_buffer,  # The true replay buffer,
            generated_data_buffer=generated_data_buffer,  # Replay buffer solely consisting of synthetic transitions
            rollout_len_func=rollout_len_func,  # Rollout length as a function of number of train calls
            rollout_len_fix=rollout_len_fix,  # fixed constant rollout length
            num_model_rollouts=num_model_rollouts,  # Number of *transitions* to generate per training timestep
            rollout_generation_freq=rollout_generation_freq,  # Can save time by only generating data when model is updated
            rollout_batch_size=rollout_batch_size,  # Maximum batch size for generating rollouts (i.e. GPU memory limit)
            real_data_pct=real_data_pct,  # Percentage of real data used for actor-critic training
            terminal_cutoff=terminal_cutoff,  # output Done if model pred > terminal_cutoff
            batch_size=batch_size,
            warm_start_epochs=warm_start_epochs,
        )

        self.adversarial_loss = None
        self.lmbda_gp = lmbda_gp

        print_banner(f"Initialized Model-based W1-GP Actor-Critic Trainer with lmbda_gp={lmbda_gp} !")

    def compute_gradient_penalty(self, network, real_samples, fake_samples, shuffle=True):
        # Calculates the gradient penalty loss for WGAN-GP
        # real_samples and fake_samples should have the same shape for interpolation
        # Random weight term for interpolation between real and fake samples

        # regularising the critic witness by constraining gradient norm to be nearly 1 along randomly chosen convex combinations of
        # generator and reference points, alpha * x_i + (1. - alpha) * y_j, for alpha ~ Uniform(0, 1)
        if shuffle:
            fake_samples = fake_samples[torch.randperm(fake_samples.shape[0])[:real_samples.shape[0]]]
        else:
            fake_samples = fake_samples[:real_samples.shape[0]]
        alpha = torch.rand(size=(real_samples.size(0), 1), device=self.device, requires_grad=False)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1. - alpha) * fake_samples).requires_grad_(True)
        dis_interpolates = network(interpolates)
        ones = torch.ones(size=(real_samples.size(0), 1), device=self.device, requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=dis_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_from_torch(self, iterations):
        self._n_epochs += 1
        real_data_pct_curr = min(max(1. - (self._n_epochs - (self.warm_start_epochs + 1)) // 5 * 0.1, self.real_data_pct), 1.)
        n_real_data = int(real_data_pct_curr * self.batch_size)
        n_generated_data = self.batch_size - n_real_data
        warm_start = self._n_epochs <= self.warm_start_epochs
        epoch_critic_loss = []

        """
        Update policy on both real and generated data
        """

        for _ in range(iterations):
            """
            Generate synthetic data using dynamics model
            """
            if self._n_train_steps_total % self.rollout_generation_freq == 0:
                generated_rewards, terminated, rollout_len = self.model_based_rollout()

            """
            Critic Training
            """
            batch = self.sample_batch(n_real_data=n_real_data, n_generated_data=n_generated_data, to_gpu=True)

            with torch.no_grad():
                # Duplicate state num_action_bellman times to sample num_action_bellman actions for each state
                next_state_repeat, next_actions = self.actor_target.sample_multiple_actions(batch['next_observations'], num_action=self.num_action_bellman, std=self.state_noise_std)
                # Compute value of the sampled actions, [Q11;Q12;Q13;Q21;Q22;Q23]
                target_Q = self.critic_target.weighted_min(next_state_repeat, next_actions, lmbda=self.lmbda)
                # find the mean Q-value over the sampled actions for each state
                target_Q = target_Q.view(self.batch_size, -1).mean(1).view(-1, 1)  # (batch_size, 1)
                target_Q = batch['rewards'].clamp(min=self.reward_min, max=self.reward_max) + (1. - batch['terminals']) * self.discount * target_Q * (target_Q.abs() < 2000.)

            current_Q1, current_Q2 = self.critic(batch['observations'], batch['actions'])
            if current_Q1.size() != target_Q.size():  # check dimensions here
                raise ValueError(f"Shape of current_Q1={current_Q1.size()}, shape of target_Q={target_Q.size()}.")

            critic_loss = F.huber_loss(current_Q1, target_Q, delta=500.) + F.huber_loss(current_Q2, target_Q, delta=500.)

            if not warm_start and critic_loss < self.epoch_critic_loss_thres:                      # do not train the critic during the warm start
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            epoch_critic_loss.append(critic_loss.detach().cpu().numpy().mean())

            """
            Actor and Discriminator Training
            """
            batch_fake_state = self.sample_batch(n_real_data=n_real_data, n_generated_data=n_generated_data, to_gpu=False)                # self.generated_data_buffer.random_batch(self.batch_size)
            batch_fake_state = batch_fake_state['observations'][np.where(batch_fake_state["terminals"] < 0.5)[0]]

            # sample noisy state-actions for each fake-state in the mini-batch
            fake_state_repeat, fake_action_samples, fake_raw_action_samples = \
                self.actor.sample_multiple_actions(torch.from_numpy(batch_fake_state).float().to(self.device),
                                                   num_action=1, std=self.state_noise_std, return_raw_action=True)
            with torch.no_grad():
                fake_transitions = self.dynamics_model.sample(torch.cat([fake_state_repeat, fake_action_samples], dim=-1))      # (batch_size, 2 + state_dim)
            if (fake_transitions != fake_transitions).any():
                fake_transitions[fake_transitions != fake_transitions] = 0
            fake_delta_obs = fake_transitions[:, 2:]
            fake_next_state = fake_state_repeat + fake_delta_obs
            fake_not_dones = (fake_transitions[:, 1] <= self.terminal_cutoff)
            fake_next_state = fake_next_state[fake_not_dones]
            _, fake_next_raw_action = self.actor(fake_next_state, return_raw_action=True)

            fake_samples = torch.cat(
                [
                    torch.cat([fake_state_repeat, fake_raw_action_samples], dim=1),             # (s_t, a_t)
                    torch.cat([fake_next_state, fake_next_raw_action], dim=1)                   # (s_{t+1}, a_{t+1})
                ], dim=0
            )
            batch_truth = self.replay_buffer.random_batch(fake_samples.shape[0])                # self.batch_size * 2
            true_samples = torch.cat(
                [   # (batch_truth['observations'], batch_truth['actions'])
                    torch.from_numpy(batch_truth['observations']).float().to(self.device), self.actor.pre_scaling_action(torch.from_numpy(batch_truth['actions']).float().to(self.device))
                ], dim=1
            )

            real_loss = self.discriminator(true_samples).mean()
            fake_loss = self.discriminator(fake_samples.detach()).mean()
            gradient_penalty = self.compute_gradient_penalty(self.discriminator, true_samples.data, fake_samples.data.detach())
            discriminator_loss = -real_loss + fake_loss + self.lmbda_gp * gradient_penalty

            generator_loss = -self.discriminator(fake_samples).mean()

            ### Compute optimization objective for policy update ###
            Q_values = self.critic.q_min(batch['observations'], self.actor(batch['observations']))
            if self.Q_average is None:
                self.Q_average = Q_values.abs().mean().detach()

            if warm_start:
                policy_loss = generator_loss
            else:
                lagrange = self.log_lagrange / self.Q_average
                policy_loss = -lagrange * Q_values.mean() + generator_loss

            # Update policy (actor): minimize policy loss
            if self._n_train_steps_total % self.policy_freq == 0:
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.beta * param.data + (1. - self.beta) * target_param.data)

            self.Q_average = self.beta * Q_values.abs().mean().detach() + (1. - self.beta) * self.Q_average

            self._n_train_steps_total += 1

        print_banner(f"Training epoch: {str(self._n_epochs)}, perform warm_start training: {warm_start}", separator="*", num_star=90)
        assert len(epoch_critic_loss) == iterations, f"len(epoch_critic_loss)={len(epoch_critic_loss)}, should be {iterations}"
        self.epoch_critic_loss_thres = 0.95 * self.epoch_critic_loss_thres + 0.05 * (np.mean(epoch_critic_loss) + 3. * np.std(epoch_critic_loss))
        # Logging
        logger.record_tabular('Num Real Data', n_real_data)
        logger.record_tabular('Num Generated Data', n_generated_data)
        logger.record_tabular('Num True Samples', true_samples.shape[0])
        logger.record_dict(create_stats_ordered_dict('Q_target', target_Q.cpu().data.numpy()))
        logger.record_tabular('Critic Loss', critic_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Epoch Critic Loss', np.array(epoch_critic_loss)))
        logger.record_tabular('Epoch Critic Loss Thres', self.epoch_critic_loss_thres)
        logger.record_tabular('Actor Loss', policy_loss.cpu().data.numpy())
        logger.record_tabular('Generator Loss', generator_loss.cpu().data.numpy())
        logger.record_tabular('Q(s,a_sample)', Q_values.mean().cpu().data.numpy())
        logger.record_tabular('Q Average', self.Q_average.cpu().data.numpy())
        logger.record_tabular('Real Loss', real_loss.cpu().data.numpy())
        logger.record_tabular('Fake Loss', fake_loss.cpu().data.numpy())
        logger.record_tabular('Gradient Penalty', gradient_penalty.cpu().data.numpy())
        logger.record_tabular('Discriminator Loss', discriminator_loss.cpu().data.numpy())
        logger.record_dict(create_stats_ordered_dict('Current_Q1', current_Q1.cpu().data.numpy()))
        logger.record_tabular('Rollout Length', rollout_len)
        logger.record_dict(create_stats_ordered_dict('Model Reward Predictions', generated_rewards))
        logger.record_tabular('Model Rollout Terminations', np.mean(terminated))
