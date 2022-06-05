import argparse
import os
from toy_networks import ImplicitPolicy, DeterministicPolicy, GaussianPolicy, DiscriminatorWithSigmoid
from GAN_Trainer import GANTrainer
from data_creation import CircleDatasetWithOneX
import utils_toy

BATCH_SIZE = 256    # Batch size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str)  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument('--log_dir', default='./results/', type=str)  # Logging directory
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--model', default='cgan', type=str)  # Logging directory
    args = parser.parse_args()

    if args.model == "cgan":
        generator = ImplicitPolicy(device=args.device)
        file_name = f"CGAN"
    elif args.model == "dgan":
        generator = DeterministicPolicy(device=args.device)
        file_name = f"DGAN"
    elif args.model == "ggan":
        generator = GaussianPolicy(device=args.device)
        file_name = f"GGAN"
    else:
        raise NotImplementedError(f"--model must be in {'cgan', 'dgan', 'ggan'}")
    generator = generator.to(args.device)
    file_name += f"-seed{args.seed}"

    folder_name = os.path.join(args.log_dir, file_name)
    print(f"File name: {file_name}, Saving location: {folder_name}", flush=True)

    if os.path.exists(folder_name):
        print(f"Remove {folder_name}")
        os.system(f"rm -rf {folder_name}")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    utils_toy.set_random_seed(args.seed)
    x_dict = {}
    noise_dict = {"noise_type": "norm", "loc": 0, "scale": 0.05}
    dataset = CircleDatasetWithOneX(r=4, n_samples=100000, seed=args.seed, batch_size=BATCH_SIZE, x_dict=x_dict, noise_dict=noise_dict)
    dataset.create_train_test_dataset(train_ratio=0.05)
    print(f"Finish creating the dataset, dimension of X and Y are {dataset.x_samples.shape} and {dataset.y.shape}")

    discriminator = DiscriminatorWithSigmoid().to(args.device)
    gan_trainer = GANTrainer(generator=generator, discriminator=discriminator, device=args.device, save_plot_loc=folder_name)
    gan_trainer.train(dataset=dataset)
