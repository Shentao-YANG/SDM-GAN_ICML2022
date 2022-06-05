import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import utils_toy
import torch
import abc


class Dataset(object):
    def __init__(self, seed, n_samples, batch_size):
        self.seed = seed
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.eps_samples = None
        utils_toy.set_random_seed(self.seed)

    @abc.abstractmethod
    def create_train_test_dataset(self):
        pass

    def create_noises(self, noise_dict):
        """
        Ref: https://pytorch.org/docs/stable/distributions.html
        :param noise_dict: {"noise_type": "norm", "loc": 0., "scale": 1.}
        """
        print("Create noises using the following parameters:")
        print(noise_dict)
        noise_type = noise_dict.get("noise_type", "norm")
        if noise_type == "t":
            dist = torch.distributions.studentT.StudentT(df=noise_dict.get("df", 10.), loc=noise_dict.get("loc", 0.0), scale=noise_dict.get("scale", 1.0))
        elif noise_type == "unif":
            dist = torch.distributions.uniform.Uniform(low=noise_dict.get("low", 0.), high=noise_dict.get("high", 1.))
        elif noise_type == "Chi2":
            dist = torch.distributions.chi2.Chi2(df=noise_dict.get("df", 10.))
        elif noise_type == "Laplace":
            dist = torch.distributions.laplace.Laplace(loc=noise_dict.get("loc", 0.), scale=noise_dict.get("scale", 1.))
        else:   # noise_type == "norm"
            dist = torch.distributions.normal.Normal(loc=noise_dict.get("loc", 0.), scale=noise_dict.get("scale", 1.))

        self.eps_samples = dist.sample((self.n_samples, 1))


class DatasetWithOneX(Dataset):
    def __init__(self, n_samples, seed, batch_size, x_dict, noise_dict):
        super(DatasetWithOneX, self).__init__(seed=seed, n_samples=n_samples, batch_size=batch_size)

        self.x_dict = x_dict
        self.x_samples = self.sample_x(self.x_dict)
        self.dim_x = self.x_samples.shape[1]  # dimension of data input
        self.y = self.create_y_from_one_x(noise_dict=noise_dict)
        self.dim_y = self.y.shape[1]  # dimension of regression output
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.train_dataset, self.train_loader = None, None

    def sample_x(self, x_dict):
        """
        :param x_dict: {"dist_type": "unif", "low": 0., "high": 1.}
        """
        print("Create x using the following parameters:")
        print(x_dict)
        dist_type = x_dict.get("dist_type", "unif")
        if dist_type == "norm":
            dist = torch.distributions.normal.Normal(loc=x_dict.get("loc", 0.), scale=x_dict.get("scale", 1.))
        else:
            dist = torch.distributions.uniform.Uniform(low=x_dict.get("low", 0.), high=x_dict.get("high", 1.))

        return dist.sample((self.n_samples, 1))

    def create_y_from_one_x(self, noise_dict):
        if self.eps_samples is None:
            n_samples_temp = self.n_samples
            if type(noise_dict.get("scale", 1.)) == torch.Tensor:
                self.n_samples = 1
            self.create_noises(noise_dict)
            if self.n_samples == 1:
                self.eps_samples = self.eps_samples[0]
                self.n_samples = n_samples_temp

    def create_train_test_dataset(self, train_ratio=0.8):
        utils_toy.set_random_seed(self.seed)
        data_idx = np.arange(self.n_samples)
        np.random.shuffle(data_idx)
        train_size = int(self.n_samples*train_ratio)
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.x_samples[data_idx[:train_size]], self.y[data_idx[:train_size]], \
            self.x_samples[data_idx[train_size:]], self.y[data_idx[train_size:]]
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(dataset=self.train_dataset,  batch_size=self.batch_size, shuffle=True)


class CircleDatasetWithOneX(DatasetWithOneX):
    def __init__(self, r, n_samples, seed, batch_size, x_dict, noise_dict):
        self.r = r
        self.theta = torch.rand((n_samples, 1)) * 2 * np.pi
        x_dict.update(r=r)
        super(CircleDatasetWithOneX, self).__init__(
            n_samples=n_samples, seed=seed, batch_size=batch_size, x_dict=x_dict, noise_dict=noise_dict
        )

    def create_y_from_one_x(self, noise_dict):
        super().create_y_from_one_x(noise_dict)
        r_samples = self.r + self.eps_samples
        self.x_samples, y = r_samples * torch.cos(self.theta), r_samples * torch.sin(self.theta)
        self.dim_x = self.x_samples.shape[1]
        return y
