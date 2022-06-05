import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import print_banner
import abc

NEGATIVE_SLOPE = 1. / 100.
print_banner(f"Negative_slope = {NEGATIVE_SLOPE}")
ATANH_MAX = 1. - 1e-7
ATANH_MIN = -1. + 1e-7


class Noise(object, metaclass=abc.ABCMeta):
    def __init__(self, device):
        self.device = device

    @abc.abstractmethod
    def sample_noise(self, shape, dtype=None, requires_grad=False):
        pass


class NormalNoise(Noise):
    def __init__(self, device, mean=0., std=1.):
        super().__init__(device=device)

        self.mean = mean
        self.std = std
        print_banner(f"Use Normal Noise with mean={self.mean} and std={self.std}.")

    def sample_noise(self, shape, dtype=None, requires_grad=False):
        return torch.randn(size=shape, dtype=dtype, device=self.device, requires_grad=requires_grad) * self.std + self.mean


class UniformNoise(Noise):
    def __init__(self, device, lower=0., upper=1.):
        super().__init__(device=device)

        self.lower = lower
        self.upper = upper
        print_banner(f"Use Uniform Noise in [{self.lower}, {self.upper}).")

    def sample_noise(self, shape, dtype=None, requires_grad=False):
        return torch.rand(size=shape, dtype=dtype, device=self.device, requires_grad=requires_grad) * (self.upper - self.lower) + self.lower


class ImplicitPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, noise, noise_method, noise_dim, device):
        # noise : {NormalNoise, UniformNoise}
        # noise_method : {"concat", "add", "multiply"}
        # noise_dim : dimension of noise for "concat" method
        super(ImplicitPolicy, self).__init__()

        self.hidden_size = (400, 300)

        if noise_dim < 1:
            noise_dim = min(10, state_dim // 2)
        noise_dim = int(noise_dim)

        print_banner(f"In implicit policy, use noise_method={noise_method} and noise_dim={noise_dim}")

        if noise_method == "concat":
            self.l1 = nn.Linear(state_dim + noise_dim, self.hidden_size[0])
        else:
            self.l1 = nn.Linear(state_dim, self.hidden_size[0])

        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], action_dim)

        self.max_action = max_action
        self.noise = noise
        self.noise_method = noise_method
        self.noise_dim = noise_dim
        self.device = device

    def forward(self, state, return_raw_action=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        # state.shape = (batch_size, state_dim)
        if self.noise_method == "concat":
            epsilon = self.noise.sample_noise(shape=(state.shape[0], self.noise_dim)).clamp(-3, 3)
            state = torch.cat([state, epsilon], 1)      # dim = (state.shape[0], state_dim + noise_dim)
        if self.noise_method == "add":
            epsilon = self.noise.sample_noise(shape=(state.shape[0], state.shape[1]))
            state = state + epsilon                     # dim = (state.shape[0], state_dim)
        if self.noise_method == "multiply":
            epsilon = self.noise.sample_noise(shape=(state.shape[0], state.shape[1]))
            state = state * epsilon                     # dim = (state.shape[0], state_dim)

        a = F.leaky_relu(self.l1(state), negative_slope=NEGATIVE_SLOPE)
        a = F.leaky_relu(self.l2(a), negative_slope=NEGATIVE_SLOPE)
        raw_actions = self.l3(a)
        if return_raw_action:
            return self.max_action * torch.tanh(raw_actions), raw_actions
        else:
            return self.max_action * torch.tanh(raw_actions)

    def sample_multiple_actions(self, state, num_action=10, std=-1., return_raw_action=False):
        # num_action : number of actions to sample from policy for each state

        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        batch_size = state.shape[0]
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        if std <= 0:
            state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        else:   # std > 0
            if num_action == 1:
                noises = torch.normal(torch.zeros_like(state), torch.ones_like(state))
                state = (state + (std * noises).clamp(-0.05, 0.05)).to(self.device)
            else:
                state_noise = state.unsqueeze(1).repeat(1, num_action-1, 1)
                noises = torch.normal(torch.zeros_like(state_noise), torch.ones_like(state_noise))
                state_noise = state_noise + (std * noises).clamp(-0.05, 0.05)
                state = torch.cat((state_noise, state.unsqueeze(1)), dim=1).view((batch_size * num_action), -1).to(self.device)  # (B * num_action) x state_dim
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        if return_raw_action:
            actions, raw_actions = self.forward(state, return_raw_action=return_raw_action)
            return state, actions, raw_actions
        else:
            return state, self.forward(state)

    def pre_scaling_action(self, actions):
        # action = self.max_action * torch.tanh(pre_tanh_action)
        # atanh(action / self.max_action) = atanh( tanh(pre_tanh_action) ) = pre_tanh_action
        return torch.atanh(torch.clamp(actions / self.max_action, min=ATANH_MIN, max=ATANH_MAX))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.hidden_size = (400, 300)

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], 1)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[0])
        self.l5 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l6 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q1 = F.leaky_relu(self.l1(state_action), negative_slope=NEGATIVE_SLOPE)
        q1 = F.leaky_relu(self.l2(q1), negative_slope=NEGATIVE_SLOPE)
        q1 = self.l3(q1)

        q2 = F.leaky_relu(self.l4(state_action), negative_slope=NEGATIVE_SLOPE)
        q2 = F.leaky_relu(self.l5(q2), negative_slope=NEGATIVE_SLOPE)
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.leaky_relu(self.l1(torch.cat([state, action], 1)), negative_slope=NEGATIVE_SLOPE)
        q1 = F.leaky_relu(self.l2(q1), negative_slope=NEGATIVE_SLOPE)
        q1 = self.l3(q1)
        return q1

    def q_min(self, state, action, no_grad=False):
        if no_grad:
            with torch.no_grad():
                q1, q2 = self.forward(state, action)
        else:
            q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def weighted_min(self, state, action, lmbda=0.75, no_grad=False):
        # lmbda * Q_min + (1-lmbda) * Q_max
        if no_grad:
            with torch.no_grad():
                q1, q2 = self.forward(state, action)
        else:
            q1, q2 = self.forward(state, action)
        return lmbda * torch.min(q1, q2) + (1. - lmbda) * torch.max(q1, q2)


class DiscriminatorWithSigmoid(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorWithSigmoid, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class DiscriminatorWithoutSigmoid(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorWithoutSigmoid, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1)
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
