import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_models.parallelized_ensemble import ParallelizedEnsemble
from utils.utils import print_banner
from copy import deepcopy


class ProbabilisticEnsemble(ParallelizedEnsemble):

    """
    Probabilistic ensemble (Chua et al. 2018).
    Implementation is parallelized such that every model uses one forward call.
    Each member predicts the mean and variance of the next state.
    Sampling is done either uniformly or via trajectory sampling.
    """

    def __init__(
            self,
            ensemble_size,        # Number of members in ensemble
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_sizes,         # Hidden sizes for each model
            device,
            spectral_norm=False,  # Apply spectral norm to every hidden layer
            separate_mean_var=True,
            pos_weight=100.,
            **kwargs
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=(obs_dim + 2) if separate_mean_var else 2 * (obs_dim + 2) - 1,      # We predict (reward, done, next_state - state), the var of 'done' is not modelled
            device=device,
            hidden_activation=nn.SiLU(),
            spectral_norm=spectral_norm,
            separate_mean_var=separate_mean_var,
            **kwargs
        )

        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.output_size = obs_dim + 2
        self.device = device
        self.separate_mean_var = separate_mean_var
        self.hidden_sizes = hidden_sizes
        self.classification_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight], device=self.device))

        self.max_logstd = nn.Parameter(
            torch.ones((1, obs_dim + 1), device=device) * (1. / 4.), requires_grad=True)
        self.min_logstd = nn.Parameter(
            -torch.ones((1, obs_dim + 1), device=device) * 5., requires_grad=True)

        print_banner(f"Initialize ProbabilisticEnsemble with ensemble_size={ensemble_size}, hidden_sizes={hidden_sizes}, spectral_norm={spectral_norm}, separate_mean_var={self.separate_mean_var}, pos_weight={pos_weight:.3f}")

    def forward(self, inputs, deterministic=False, return_dist=False):
        output = super().forward(inputs)
        if not self.separate_mean_var:
            mean, logstd = torch.chunk(output, 2, dim=-1)
        else:       # separate_mean_var
            mean, logstd = output

        # Variance clamping to prevent poor numerical predictions
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)

        if deterministic:
            return mean, logstd if return_dist else mean

        std = torch.exp(logstd)
        std = torch.cat([std[..., :1],  torch.zeros(std[..., :1].shape, device=self.device), std[..., 1:]], dim=-1)
        eps = torch.randn(std.shape, device=self.device)
        samples = mean + std * eps

        if return_dist:
            return samples, mean, logstd
        else:
            return samples

    def get_loss(self, x, y, split_by_model=False, return_l2_error=False):
        # Note: we assume y here already accounts for the delta of the next state

        mean, logstd = self.forward(x, deterministic=True, return_dist=True)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        mean_rns = mean[..., self.rns_torch]
        mean_d = mean[..., 1:2]
        y_rns = y[..., self.rns_torch]
        y_d = y[..., 1:2]

        # Maximize log-probability of transitions
        inv_var = torch.exp(-2 * logstd)
        sq_l2_error = (mean_rns - y_rns) ** 2
        termination_loss = self.classification_loss(mean_d, y_d).mean(dim=-1).mean(dim=-1)
        if return_l2_error:
            l2_error = sq_l2_error.mean(dim=-1).mean(dim=-1) + termination_loss

        loss = (sq_l2_error * inv_var + 2 * logstd).mean(dim=-1).mean(dim=-1) + termination_loss

        if split_by_model:
            losses = [loss[i] for i in range(self.ensemble_size)]
            if return_l2_error:
                l2_errors = [l2_error[i] for i in range(self.ensemble_size)]
                return losses, l2_errors
            else:
                return losses
        else:   # return the loss averaged over all the ensemble models
            clipping_bound_loss = 0.01 * (self.max_logstd * 2).sum() - 0.01 * (self.min_logstd * 2).sum()
            if return_l2_error:
                return loss.sum() + clipping_bound_loss, l2_error.sum()
            else:
                return loss.sum() + clipping_bound_loss

    @property
    def num_hidden_layers(self):
        return len(self.hidden_sizes)

    def get_idv_model_state(self, idx):
        params = [{"W": deepcopy(self.fcs[i].W.data[idx]), "b": deepcopy(self.fcs[i].b.data[idx])} for i in range(self.num_hidden_layers)]
        params.append({"W": deepcopy(self.last_fc.W.data[idx]), "b": deepcopy(self.last_fc.b.data[idx])})
        if self.separate_mean_var:
            params.append({"W": deepcopy(self.last_fc_std.W.data[idx]), 'b': deepcopy(self.last_fc_std.b.data[idx])})

        return params

    def load_model_state_from_dict(self, state_dict):
        num_hidden_layers = self.num_hidden_layers
        for model in range(self.ensemble_size):
            model_params = state_dict[model]
            for i in range(num_hidden_layers):
                self.fcs[i].W.data[model].copy_(model_params[i]["W"])
                self.fcs[i].b.data[model].copy_(model_params[i]['b'])
            self.last_fc.W.data[model].copy_(model_params[num_hidden_layers]["W"])
            self.last_fc.b.data[model].copy_(model_params[num_hidden_layers]['b'])
            if self.separate_mean_var:
                self.last_fc_std.W.data[model].copy_(model_params[num_hidden_layers+1]["W"])
                self.last_fc_std.b.data[model].copy_(model_params[num_hidden_layers+1]["b"])
