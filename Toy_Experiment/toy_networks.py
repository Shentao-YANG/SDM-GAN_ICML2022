import numpy as np
import torch
import torch.nn as nn

DIM = 128   # Model dimensionality


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class NormalNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        print(f"Use Normal Noise with mean={self.mean} and std={self.std}.")

    def sample_noise(self, shape, device, dtype=None, requires_grad=False):
        return torch.randn(size=shape, dtype=dtype, device=device, requires_grad=requires_grad) * self.std + self.mean


class ImplicitPolicy(nn.Module):
    def __init__(self, device, state_dim=1, action_dim=1, noise_dim=2):
        super(ImplicitPolicy, self).__init__()
        self.noise = NormalNoise()
        self.model = nn.Sequential(
            nn.Linear(state_dim + noise_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, action_dim)
        )
        self.model.apply(init_weights)
        self.noise_dim = noise_dim
        self.device = device

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        epsilon = self.noise.sample_noise(shape=(state.shape[0], self.noise_dim), device=self.device)
        state = torch.cat([state, epsilon], 1)      # dim = (state.shape[0], state_dim + noise_dim)

        return self.model(state)

    def sample_multiple_actions(self, state, num_action=10):
        # num_action : number of actions to sample from policy for each state
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        return state, self.forward(state)


class DeterministicPolicy(nn.Module):
    def __init__(self, device, state_dim=1, action_dim=1):
        super(DeterministicPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, action_dim)
        )
        self.model.apply(init_weights)
        self.device = device

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        # state.shape = (batch_size, state_dim)
        return self.model(state)

    def sample_multiple_actions(self, state, num_action=10):
        # num_action : number of actions to sample from policy for each state
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        return state, self.forward(state)


class GaussianPolicy(nn.Module):
    def __init__(self, device, state_dim=1, action_dim=1):
        super(GaussianPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU()
        )
        self.mean = nn.Linear(DIM, action_dim)
        self.logstd = nn.Linear(DIM, action_dim)
        self.model.apply(init_weights)
        self.mean.apply(init_weights)
        self.logstd.apply(init_weights)
        self.device = device

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        h = self.model(state)
        a_mean = self.mean(h)
        a_std = torch.exp(self.logstd(h).clamp(-5., 5.))
        a = a_mean + a_std * torch.randn_like(a_std).to(self.device)

        return a

    def sample_multiple_actions(self, state, num_action=10):
        # num_action : number of actions to sample from policy for each state
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        # e.g., num_action = 3, [s1;s2] -> [s1;s1;s1;s2;s2;s2]
        state = state.unsqueeze(1).repeat(1, num_action, 1).view(-1, state.size(-1)).to(self.device)
        # return [a11;a12;a13;a21;a22;a23] for [s1;s1;s1;s2;s2;s2]
        return state, self.forward(state)


class DiscriminatorWithSigmoid(nn.Module):
    def __init__(self, state_dim=1, action_dim=1):
        super(DiscriminatorWithSigmoid, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, 1),
            nn.Sigmoid()
        )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class DiscriminatorWithoutSigmoid(nn.Module):
    def __init__(self, state_dim=1, action_dim=1):
        super(DiscriminatorWithoutSigmoid, self).__init__()
        self.hidden_size = (400, 300)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM, 1)
        )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
