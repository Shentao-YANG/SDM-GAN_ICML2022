import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
import os


CRITIC_ITERS = 5    # How many critic iterations per generator iteration
ITERS = 100000      # how many generator iterations to train for
PLOTITERS = ITERS // 10


class GANTrainer(object):
    def __init__(self, generator, discriminator, device, save_plot_loc):
        self.generator = generator
        self.discriminator = discriminator

        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

        self.class_loss = nn.BCELoss()
        self.device = device
        self.save_plot_loc = save_plot_loc

    def train(self, dataset):
        true_labels = 0.9 * torch.ones(dataset.batch_size, 1, device=self.device)
        fake_labels = torch.zeros(dataset.batch_size, 1, device=self.device)
        generator_true_labels = torch.ones(dataset.batch_size, 1, device=self.device)

        self.plot_test_set(dataset=dataset)

        for iteration in range(1, ITERS+1):
            x_batch, y_batch = next(iter(dataset.train_loader))
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            fake_samples = torch.cat([x_batch, self.generator(x_batch)], dim=1)             # (s_t, a_t)

            if iteration > 1:   # Train generator
                generator_loss = self.class_loss(self.discriminator(fake_samples), generator_true_labels)
                self.gen_optim.zero_grad()
                generator_loss.backward()
                self.gen_optim.step()

            for _ in range(CRITIC_ITERS):  # Train critic
                x_batch, y_batch = next(iter(dataset.train_loader))
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                true_samples = torch.cat([x_batch, y_batch], dim=1)
                fake_samples = torch.cat([x_batch, self.generator(x_batch)], dim=1)

                real_loss = self.class_loss(self.discriminator(true_samples), true_labels)
                fake_loss = self.class_loss(self.discriminator(fake_samples.detach()), fake_labels)
                discriminator_loss = (real_loss + fake_loss) / 2.

                self.disc_optim.zero_grad()
                discriminator_loss.backward()
                self.disc_optim.step()

            if iteration % PLOTITERS == 0:
                print(f"Finish {iteration} iterations")
                self.plot_test_result(dataset=dataset, iteration=iteration)

    def plot_test_result(self, dataset, iteration):
        self.generator.eval()
        with torch.no_grad():
            x_instances, y_instances = dataset.x_test.clone(), dataset.y_test.clone()
            generated_y = self.generator(x_instances.to(self.device)).data.cpu().numpy().ravel()
        x_instances = x_instances.data.cpu().numpy().ravel()
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))
        sns.kdeplot(x_instances, generated_y, ax=ax, cmap="Blues", shade=True, bw=0.1)
        ax.set_xlim(-dataset.r * 1.2, dataset.r * 1.2)
        ax.set_ylim(-dataset.r * 1.2, dataset.r * 1.2)
        fig.savefig(os.path.join(self.save_plot_loc, 'iter_{}.pdf'.format(int(iteration))))
        plt.close()

    def plot_test_set(self, dataset):
        x_instances, y_instances = dataset.x_test.clone().data.cpu().numpy().ravel(), dataset.y_test.clone().data.cpu().numpy().ravel()
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))
        sns.kdeplot(x_instances, y_instances, ax=ax, cmap="Blues", shade=True, bw=0.1)
        ax.set_xlim(-dataset.r * 1.2, dataset.r * 1.2)
        ax.set_ylim(-dataset.r * 1.2, dataset.r * 1.2)
        fig.savefig(os.path.join(self.save_plot_loc, 'true_testset.pdf'))
        plt.close()
