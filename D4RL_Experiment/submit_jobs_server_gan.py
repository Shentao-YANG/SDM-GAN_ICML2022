import argparse
import os


def write_front(args, count):
    print(f"The system is {args.sys}")
    if args.sys == "server":
        front = f"""#!/bin/bash


"""
    return front


if __name__ == "__main__":

    envs_list = [
        # (env_name, fixed_rollout_len, noise_dim)
        ('halfcheetah-medium-expert-v0', 3, 0),
        ('hopper-medium-expert-v0', 5, 0),
        ('walker2d-medium-expert-v0', 5, 50),
        ('halfcheetah-medium-v0', 3, 0),
        ('hopper-medium-v0', 1, 50),
        ('walker2d-medium-v0', 1, 0),
        ('halfcheetah-medium-replay-v0', 3, 50),
        ('hopper-medium-replay-v0', 1, 50),
        ('walker2d-medium-replay-v0', 5, 50),
        ('maze2d-umaze-v1', 3, 0),
        ('maze2d-medium-v1', 3, 50),
        ('maze2d-large-v1', 3, 50),
        ('pen-cloned-v0', 1, 0),
        ('pen-expert-v0', 3, 50),
        ('door-expert-v0', 3, 0),
        ('pen-human-v0', 3, 0),
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", type=str, default='server')

    parser.add_argument("--num_seed", type=int, default=3)                                              #
    parser.add_argument("--submit_job", type=str, default="False")                                      #
    parser.add_argument("--num_thread", default=8, type=int)                                            #
    parser.add_argument("--device", type=str, default="0")                                              # for multi-device server, use e.g. "0,1,2,3"

    parser.add_argument("--ExpID", default=1, type=int)                                                 # Experiment ID
    parser.add_argument("--total_iters", default=1e6, type=float)                                       # total number of iterations for MB-SGD training

    parser.add_argument('--actor_lr', default=2e-4, type=float)                                         # actor learning rate
    parser.add_argument('--critic_lr', default=3e-4, type=float)                                        # critic learning rate
    parser.add_argument('--dis_lr', default=2e-4, type=float)                                           # act navigator learning rate
    parser.add_argument('--model_lr', default=0.001, type=float)                                        # act navigator learning rate (was: 3e-4)
    parser.add_argument('--batch_size', default=512, type=int)                                          # batch size of MB-SGD
    parser.add_argument('--log_lagrange', default=10.0, type=float)                                      # maximum value of log lagrange multiplier
    parser.add_argument('--log_lagrange_am', default=10.0, type=float)                                   # 10.0 maximum value of log lagrange multiplier
    parser.add_argument('--log_lagrange_human', default=10.0, type=float)                                # 15.0 maximum value of log lagrange multiplier
    parser.add_argument('--policy_freq', default=2, type=int)                                           # update frequency of the actor

    parser.add_argument('--noise_type', default='normal', type=str)                                     # noise type
    parser.add_argument('--noise_method', default="concat", type=str)                                   # method to add noise in the implicit policy
    parser.add_argument("--noise_dim", default=0, type=int)                                             # dimension of noise in the implicit policy
    parser.add_argument('--sigma', default=1.0, type=float)                                             # standard deviation of the normal noise
    parser.add_argument('--sigma_human', default=1.0, type=float)                                       # 1e-3 standard deviation of the normal noise
    parser.add_argument('--lower', default=0.0, type=float)                                             # lower bound of uniform noise
    parser.add_argument('--upper', default=1.0, type=float)                                             # upper bound of uniform noise
    parser.add_argument('--warm_start_epochs', default=40, type=int)                                    # number of epochs for warm start training
    parser.add_argument('--state_noise_std', default=3e-4, type=float)
    parser.add_argument('--num_action_bellman', default=50, type=int)
    parser.add_argument('--rollout_generation_freq', default=250, type=int)
    parser.add_argument('--rollout_batch_size', default=2048, type=int)
    parser.add_argument('--num_model_rollouts', default=128, type=int)
    parser.add_argument('--rollout_retain_epochs', default=5, type=int)
    parser.add_argument("--model_spectral_norm", default='False', type=str)                             # Whether apply spectral norm to every hidden layer (default: False)
    parser.add_argument('--fixed_rollout_len', default=1, type=int)
    parser.add_argument('--real_data_pct', default=0.5, type=float)
    parser.add_argument('--terminal_cutoff', default=0., type=float)                                   # Use a very large value basically means always not_done <=> do not model termination condition
    parser.add_argument('--model_epochs_since_last_update', default=10, type=int)
    args = parser.parse_args()

    if not os.path.exists("./job_scripts"):
        os.makedirs("./job_scripts")
    if not os.path.exists(f"./python_outputs/Exp{args.ExpID}"):
        os.makedirs(f"./python_outputs/Exp{args.ExpID}")

    job_list = []
    gpu_devive = args.device.split(",")
    print(f"Device: {gpu_devive}")
    for idx, (env_name, fixed_rollout_len, noise_dim) in enumerate(envs_list):
        log_lagrange = args.log_lagrange
        sigma = args.sigma

        if (('antmaze' in env_name) or ('cloned' in env_name) or ('expert' in env_name)) and ("medium" not in env_name):
            log_lagrange = args.log_lagrange_am
            print(f"{env_name}: log_lagrange={log_lagrange}")
        if 'human' in env_name:
            sigma = args.sigma_human
            log_lagrange = args.log_lagrange_human
            print(f"{env_name}: sigma={sigma}, log_lagrange={log_lagrange}")

        if idx % len(gpu_devive) == 0:
            if len(gpu_devive) == 1:
                job_file_name = f"./job_scripts/run_Exp{args.ExpID}_{env_name}.sh"
            else:
                job_file_name = f"./job_scripts/run_Exp{args.ExpID}_{idx // len(gpu_devive) + 1}.sh"
            job_list.append(job_file_name)
            f = open(job_file_name, "w")
            f.write(write_front(args=args, count=(idx // len(gpu_devive) + 1)))
            run_cmd = f"for ((i=0;i<{args.num_seed};i+=1)) \n"
            run_cmd += "do \n"

        run_cmd += f"  OMP_NUM_THREADS={args.num_thread} MKL_NUM_THREADS={args.num_thread} python gan_main.py \\\n"
        run_cmd += f"  --device 'cuda:{gpu_devive[idx % len(gpu_devive)]}' \\\n"
        run_cmd += f'  --log_dir "./results/" \\\n'
        run_cmd += f"  --batch_size {args.batch_size} \\\n"
        run_cmd += f'  --env_name "{env_name}" \\\n'
        run_cmd += f"  --policy_freq {args.policy_freq} \\\n"
        run_cmd += f"  --sigma {sigma} \\\n"
        run_cmd += '  --seed $i \\\n'
        run_cmd += f"  --total_iters {args.total_iters} \\\n"
        run_cmd += f"  --dis_lr {args.dis_lr} \\\n"
        run_cmd += f"  --actor_lr {args.actor_lr} \\\n"
        run_cmd += f"  --critic_lr {args.critic_lr} \\\n"
        run_cmd += f"  --model_lr {args.model_lr} \\\n"
        run_cmd += f"  --log_lagrange {log_lagrange} \\\n"
        run_cmd += f"  --noise_dim {noise_dim} \\\n"
        run_cmd += f"  --noise_type {args.noise_type} \\\n"
        run_cmd += f"  --noise_method {args.noise_method} \\\n"
        run_cmd += f"  --lower {args.lower} \\\n"
        run_cmd += f"  --upper {args.upper} \\\n"
        run_cmd += f"  --warm_start_epochs {args.warm_start_epochs} \\\n"
        run_cmd += f"  --state_noise_std {args.state_noise_std} \\\n"
        run_cmd += f"  --num_action_bellman {args.num_action_bellman} \\\n"
        run_cmd += f"  --rollout_generation_freq {args.rollout_generation_freq} \\\n"
        run_cmd += f"  --rollout_batch_size {args.rollout_batch_size} \\\n"
        run_cmd += f"  --num_model_rollouts {args.num_model_rollouts} \\\n"
        run_cmd += f"  --rollout_retain_epochs {args.rollout_retain_epochs} \\\n"
        run_cmd += f"  --model_spectral_norm {args.model_spectral_norm} \\\n"
        run_cmd += f"  --fixed_rollout_len {fixed_rollout_len} \\\n"
        run_cmd += f"  --real_data_pct {args.real_data_pct} \\\n"
        run_cmd += f"  --terminal_cutoff {args.terminal_cutoff} \\\n"
        run_cmd += f"  --model_epochs_since_last_update {args.model_epochs_since_last_update} \\\n"
        run_cmd += f"  --ExpID {args.ExpID}"
        if len(gpu_devive) == 1:
            run_cmd += f' > "./python_outputs/Exp{args.ExpID}/{env_name}_$i.txt" & \n'
        else:
            run_cmd += f' > "./python_outputs/Exp{args.ExpID}/task{idx // len(gpu_devive) + 1}{idx % len(gpu_devive) + 1}_$i.txt" & \n'

        if idx % len(gpu_devive) == (len(gpu_devive) - 1):
            run_cmd += "done \n"
            run_cmd += "wait \n"
            run_cmd += "\n"
            run_cmd += "printf '\\nEnd of running...\\n'\n"
            run_cmd += "#########################################################################"

            f.write(run_cmd)
            f.close()

    if args.submit_job == "True":
        for job in job_list:
            print(f"submit job: {job}")
            os.system(f"bash {job}")
    else:
        print(*job_list, sep="\n")
