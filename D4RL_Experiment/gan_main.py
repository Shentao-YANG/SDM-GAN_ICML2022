import argparse
import gym
import numpy as np
import os
import pickle

from utils.logger import setup_logger
import d4rl
import torch
from datetime import datetime
from utils.utils import print_banner, always_n
from networks import NormalNoise, UniformNoise, ImplicitPolicy, Critic, DiscriminatorWithSigmoid
from dynamic_models.probabilistic_ensemble import ProbabilisticEnsemble
from dynamic_models.model_trainer import ModelTrainer
from actor_critic_trainer import GanACTrainer
from mb_joint_matching_trainer import GanMBJointMatchingTrainer
from replay_buffers.env_replay_buffer import EnvReplayBuffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--ExpID", default=1, type=int)                                                 # Experiment ID
    parser.add_argument('--device', default='cpu', type=str)                                            # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument('--log_dir', default='./results/', type=str)                                    # Logging directory
    parser.add_argument("--load_model", default=None, type=str)                                         # Load model and optimizer parameters
    parser.add_argument("--save_models", default='True', type=str)                                      # Save model and optimizer parameters, (default: True)
    parser.add_argument("--save_freq", default=1e6, type=int)                                           # How often we saves the model
    parser.add_argument("--env_name", default="walker2d-medium-v0", type=str)                           # OpenAI gym environment name
    parser.add_argument("--dataset", default=None, type=str)                                            # path to dataset other than d4rl env
    parser.add_argument("--seed", default=0, type=int)                                                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1000, type=int)                                          # How often (# of iterations of MB-SGD) we evaluate
    parser.add_argument("--total_iters", default=1e6, type=float)                                       # total number of iterations for MB-SGD training
    ### Optimization Setups ###
    parser.add_argument('--actor_lr', default=2e-4, type=float)                                         # actor learning rate
    parser.add_argument('--critic_lr', default=3e-4, type=float)                                        # critic learning rate
    parser.add_argument('--dis_lr', default=2e-4, type=float)                                           # act navigator learning rate
    parser.add_argument('--model_lr', default=0.001, type=float)                                        # act navigator learning rate (was: 3e-4)
    parser.add_argument('--batch_size', default=512, type=int)                                          # batch size of MB-SGD
    parser.add_argument('--log_lagrange', default=10, type=float)                                      # maximum value of log lagrange multiplier
    parser.add_argument('--policy_freq', default=2, type=int)                                           # update frequency of the actor
    ### Algorithm Specific Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)                                         # Discount factor
    parser.add_argument("--beta", default=0.005, type=float)                                            # Target network update rate
    parser.add_argument("--lmbda", default=0.75, type=float)                                            # weight of the minimum in Q-update
    parser.add_argument('--noise_type', default='normal', type=str)                                     # noise type
    parser.add_argument('--noise_method', default="concat", type=str)                                   # method to add noise in the implicit policy
    parser.add_argument("--noise_dim", default=0, type=int)                                             # dimension of noise in the implicit policy
    parser.add_argument('--mu', default=0.0, type=float)                                                # mean of the normal noise
    parser.add_argument('--sigma', default=1.0, type=float)                                             # standard deviation of the normal noise
    parser.add_argument('--lower', default=0.0, type=float)                                             # lower bound of uniform noise
    parser.add_argument('--upper', default=1.0, type=float)                                             # upper bound of uniform noise
    parser.add_argument('--warm_start_epochs', default=40, type=int)                                    # number of epochs for warm start training
    parser.add_argument('--state_noise_std', default=3e-4, type=float)
    parser.add_argument('--num_action_bellman', default=50, type=int)
    parser.add_argument('--rollout_generation_freq', default=250, type=int)
    parser.add_argument('--rollout_batch_size', default=2048, type=int)
    parser.add_argument('--num_model_rollouts', default=128, type=int)
    parser.add_argument('--rollout_retain_epochs', default=5, type=int)
    parser.add_argument('--ensemble_size', default=7, type=int)
    parser.add_argument('--num_elites', default=5, type=int)
    parser.add_argument("--model_spectral_norm", default='False', type=str)                             # Whether apply spectral norm to every hidden layer (default: False)
    parser.add_argument('--fixed_rollout_len', default=1, type=int)
    parser.add_argument('--real_data_pct', default=0.5, type=float)
    parser.add_argument('--terminal_cutoff', default=0., type=float)                                   # Use a very large value basically means always not_done <=> do not model termination condition
    parser.add_argument('--model_max_grad_steps', default=1e7, type=float)
    parser.add_argument('--model_epochs_since_last_update', default=10, type=int)
    args = parser.parse_args()

    args.save_models = args.save_models == 'True'
    args.model_spectral_norm = args.model_spectral_norm == 'True'

    if args.dataset is None:
        args.dataset = args.env_name
    args.log_dir = args.log_dir + f"Exp{args.ExpID}"
    # Setup Logging
    file_name = f"G|{args.dataset}|Exp{args.ExpID}|{args.noise_type}|{args.noise_method}|noiD{args.noise_dim}|sigma{args.sigma}|upper{args.upper}|ll{args.log_lagrange}|pf{args.policy_freq}|{args.seed}"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    print_banner(f"Saving location: {folder_name}")
    if os.path.exists(os.path.join(folder_name, 'variant.json')):
        raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version="MBGAN|Implicit Policy")

    # Setup Environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(folder_name), variant=variant, log_dir=folder_name)
    print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load Dataset
    if args.env_name == args.dataset:
        dataset = d4rl.qlearning_dataset(env)  # Load d4rl dataset
    else:
        dataset_file = os.path.dirname(os.path.abspath(__file__)) + '/dataset/' + args.dataset + '.pkl'
        dataset = pickle.load(open(dataset_file, 'rb'))
        print_banner(f"Loaded data from {dataset_file}")

    replay_buffer_offline = EnvReplayBuffer(max_replay_buffer_size=int(dataset['observations'].shape[0]), env=env, store_log_probs=False)
    print_banner("Begin Loading Offline Dataset")
    t1 = datetime.now()
    replay_buffer_offline.add_path(dataset)
    print_banner(f"Finish Loading Offline Dataset, using time {datetime.now() - t1}, top of replay_buffer_offline is: {replay_buffer_offline.top()}")
    reward_stat = replay_buffer_offline.reward_stat
    print_banner(f"Reward Info: Mean {reward_stat['mean']:.4f}, Std {reward_stat['std']:.4f}, Max {reward_stat['max']:.4f}, Min {reward_stat['min']:.4f}")
    replay_buffer_generated = EnvReplayBuffer(
        max_replay_buffer_size=int(args.rollout_generation_freq * args.num_model_rollouts * (args.eval_freq / args.rollout_generation_freq) * args.rollout_retain_epochs),
        env=env, store_log_probs=False
    )

    support_noise_type = {"normal", "uniform"}
    support_noise_methods = {"concat", "add", "multiply"}

    if args.noise_method not in support_noise_methods:
        raise NotImplementedError(f"Receive noise_method: {args.noise_method}, currently only support {support_noise_methods}")

    if args.noise_type not in support_noise_type:
        raise NotImplementedError(f"Receive noise_type: {args.noise_type}, currently only support {support_noise_type}")

    if args.noise_type == "normal":
        noise = NormalNoise(device=args.device, mean=args.mu, std=args.sigma)
    else:
        noise = UniformNoise(device=args.device, lower=args.lower, upper=args.upper)

    actor = ImplicitPolicy(state_dim, action_dim, max_action, noise, args.noise_method, args.noise_dim, args.device).to(args.device)
    critic = Critic(state_dim, action_dim).to(args.device)
    discriminator = DiscriminatorWithSigmoid(state_dim=state_dim, action_dim=action_dim).to(args.device)
    print_banner(f"state_noise_std={args.state_noise_std}, num_action_bellman={args.num_action_bellman}")
    environment_model = ProbabilisticEnsemble(
        ensemble_size=args.ensemble_size,
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=[200] * 4,
        device=args.device,
        spectral_norm=args.model_spectral_norm,
        pos_weight=min(replay_buffer_offline.terminals_pos_weight, 1e4)
    )
    environment_model_trainer = ModelTrainer(
        ensemble=environment_model,
        device=args.device,
        replay_buffer=replay_buffer_offline,
        num_elites=args.num_elites,
        learning_rate=args.model_lr,
        batch_size=int(min(args.batch_size, 128)),
        optimizer_class=torch.optim.Adam,
    )
    AC_trainer = GanACTrainer(
        device=args.device,
        discount=args.discount,
        beta=args.beta,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        dis_lr=args.dis_lr,
        lmbda=args.lmbda,
        log_lagrange=args.log_lagrange,
        policy_freq=args.policy_freq,
        state_noise_std=args.state_noise_std,
        num_action_bellman=args.num_action_bellman,
        actor=actor,
        critic=critic,
        discriminator=discriminator,
        dynamics_model=environment_model,
        replay_buffer=replay_buffer_offline,
        generated_data_buffer=replay_buffer_generated,
        rollout_len_func=always_n,
        rollout_len_fix=args.fixed_rollout_len,
        num_model_rollouts=args.num_model_rollouts,
        rollout_generation_freq=args.rollout_generation_freq,
        rollout_batch_size=args.rollout_batch_size,
        real_data_pct=args.real_data_pct,
        terminal_cutoff=args.terminal_cutoff,
        batch_size=args.batch_size,
        warm_start_epochs=args.warm_start_epochs
    )
    joint_matching_trainer = GanMBJointMatchingTrainer(
        device=args.device,
        model_trainer=environment_model_trainer,
        ac_trainer=AC_trainer,
        total_iters=args.total_iters,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        save_models=args.save_models,
        save_folder_name=folder_name,
        env=env,
        model_max_grad_steps=int(args.model_max_grad_steps),
        model_epochs_since_last_update=args.model_epochs_since_last_update
    )

    t1 = datetime.now()
    joint_matching_trainer.train()
    print_banner(f"FINISH TRAINING !!! Using time {datetime.now() - t1}")
