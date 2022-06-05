import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from utils.utils import print_banner


def identity(x):
    return x


class ParallelizedLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        device,
        w_std_value=1.0,
        b_init_value=0.0,
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = torch.randn((ensemble_size, input_dim, output_dim), device=device)
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = torch.zeros((ensemble_size, 1, output_dim), device=device).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        # output dim is: (ensemble_size, batch_size, output_dim)
        return x @ self.W + self.b


class ParallelizedEnsemble(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            device,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.0,
            spectral_norm=False,
            separate_mean_var=False
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.device = device
        self.separate_mean_var = separate_mean_var

        print_banner(f"Initialize ParallelizedEnsemble with ensemble_size={ensemble_size}, hidden_sizes={hidden_sizes}, spectral_norm={spectral_norm}, separate_mean_var={self.separate_mean_var}")

        # data normalization
        self.input_mu = nn.Parameter(
            torch.zeros(input_size, device=device), requires_grad=False).float()
        self.input_std = nn.Parameter(
            torch.ones(input_size, device=device), requires_grad=False).float()

        obs_dim = self.output_size - 2 if self.separate_mean_var else (self.output_size + 1) // 2 - 2
        self.rns_np = np.array([True, False] + [True] * int(obs_dim))
        self.rns_torch = torch.from_numpy(self.rns_np).to(self.device)

        self.delta_obs_mu = nn.Parameter(
            torch.zeros(self.rns_np.sum(), device=device), requires_grad=False).float()
        self.delta_obs_std = nn.Parameter(
            torch.ones(self.rns_np.sum(), device=device), requires_grad=False).float()

        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayer(
                ensemble_size, in_size, next_size,
                device=device,
                w_std_value=1./(2*np.sqrt(in_size)),
                b_init_value=b_init_value,
            )
            if spectral_norm:
                fc = torch.nn.utils.parametrizations.spectral_norm(fc, name='W')
            self.__setattr__('fc%d' % i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayer(
            ensemble_size, in_size, output_size,
            device=device,
            w_std_value=1./(2*np.sqrt(in_size)),
            b_init_value=b_init_value,
        )
        if self.separate_mean_var:
            self.last_fc_std = ParallelizedLayer(
                ensemble_size, in_size, output_size - 1,        # the var of 'done' is not modelled
                device=device,
                w_std_value=1. / (2 * np.sqrt(in_size)),
                b_init_value=b_init_value,
            )

    def forward(self, inputs):
        dim = len(inputs.shape)

        # inputs normalization
        h = (inputs - self.input_mu) / self.input_std

        # repeat h to make amenable to parallelization
        # if dim = 3, then we already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)

        if not self.separate_mean_var:
            preactivation = self.last_fc(h)
            output = self.output_activation(preactivation)
        else:       # separate_mean_var
            preactivation, preactivation_std = self.last_fc(h), self.last_fc_std(h)
            output = self.output_activation(preactivation), self.output_activation(preactivation_std)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            if not self.separate_mean_var:
                output = output.squeeze(1)
            else:       # separate_mean_var
                output = output[0].squeeze(1), output[1].squeeze(1)

        # output is (ensemble_size, batch_size, output_size) or tuple (mean, logstd) if separate_mean_var
        return output

    def sample(self, inputs):
        preds = self.forward(inputs)
        batch_size = preds.shape[1]
        model_idxes = np.random.choice(self.elites, size=batch_size)
        batch_idxes = np.arange(0, batch_size)
        samples = preds[model_idxes, batch_idxes]

        # return unnormalized delta
        samples[..., self.rns_torch] = samples[..., self.rns_torch] * self.delta_obs_std + self.delta_obs_mu
        return samples

    def fit_input_stats(self, data, y=None, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std != std] = 0
        std[std < 1e-12] = 1e-12
        if y is not None:
            delta_mean = np.mean(y[:, self.rns_np], axis=0, keepdims=True)
            delta_std = np.std(y[:, self.rns_np], axis=0, keepdims=True)
            delta_std[delta_std != delta_std] = 0
            delta_std[delta_std < 1e-12] = 1e-12

            self.delta_obs_mu.data = torch.from_numpy(delta_mean).float().to(self.device)
            self.delta_obs_std.data = torch.from_numpy(delta_std).float().to(self.device)

        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = torch.from_numpy(mean).float().to(self.device)
        self.input_std.data = torch.from_numpy(std).float().to(self.device)
