from utils.utils import print_banner
from utils.logger import logger
import torch
import numpy as np
import os
from datetime import datetime


class GanMBJointMatchingTrainer(object):
    def __init__(
            self,
            device,
            model_trainer,
            ac_trainer,
            total_iters,
            eval_freq,
            save_freq,
            save_models,
            save_folder_name,
            env,
            model_max_grad_steps=int(1e7),          # The model will train until either this number of grad steps
            model_epochs_since_last_update=10,      # or until holdout loss converged for this number of epochs
    ):
        self.device = device
        self.model_trainer = model_trainer
        self.ac_trainer = ac_trainer
        self.total_iters = total_iters
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_models = save_models
        self.env = env
        self.save_folder_name = save_folder_name
        self.model_max_grad_steps = model_max_grad_steps
        self.model_epochs_since_last_update = model_epochs_since_last_update

        print_banner("Initialized Model-based GAN Actor-Critic with Implicit Policy!")

    def select_action(self, state):
        # For a given state, sample 10 actions and return the one with highest Q_1 value
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(self.device)
            action = self.ac_trainer.actor(state)
            q1 = self.ac_trainer.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def eval_policy(self, eval_episodes=10):
        # Run policy for eval_episodes episodes and returns average undiscounted episodic reward
        avg_reward = 0.0
        for _ in range(eval_episodes):
            state, done = self.env.reset(), False
            while not done:
                action = self.select_action(np.array(state))
                state, reward, done, _ = self.env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        info = {'Average Episodic Reward': avg_reward}
        print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        return info

    def train(self):
        # Pretrain the model at the beginning of training until convergence
        # Note that convergence is measured against a holdout set of max size 8192
        t1 = datetime.now()
        self.model_trainer.train_from_buffer(
            holdout_pct=0.2,
            max_grad_steps=self.model_max_grad_steps,
            epochs_since_last_update=self.model_epochs_since_last_update,
        )
        print_banner(f"FINISH TRAINING MODEL !!! Using time {datetime.now() - t1}", separator="*", num_star=90)

        evaluations = []
        while self.ac_trainer.num_train_steps < self.total_iters:
            logger.record_tabular('Training Epochs', self.ac_trainer.num_epochs + 1)
            self.ac_trainer.train_from_torch(iterations=int(self.eval_freq))

            # Save Model
            if self.ac_trainer.num_train_steps % self.save_freq == 0 and self.save_models:
                self.ac_trainer.save('model_%i' % self.ac_trainer.num_train_steps, self.save_folder_name)

            info = self.eval_policy()
            evaluations.append(info['Average Episodic Reward'])
            np.save(
                os.path.join(self.save_folder_name, 'eval_norm'),
                self.env.get_normalized_score(np.array(evaluations)) * 100.0
            )
            np.save(os.path.join(self.save_folder_name, 'eval'), evaluations)
            for k, v in info.items():
                logger.record_tabular(k, v)

            logger.dump_tabular(with_timestamp=False)

            if self.ac_trainer.num_epochs % 100 == 0:
                res_dict = dict()
                for last in [1, 5, 10, 20, 50, 100]:
                    res_dict[f"Last {str(last)}"] = np.mean(evaluations[-last:]).round(1)
                print(res_dict)

    def get_snapshot(self):
        return self.ac_trainer.get_snapshot()
