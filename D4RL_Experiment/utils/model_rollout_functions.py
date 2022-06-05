import torch
from utils.utils import get_numpy


def _create_full_tensors(start_states, max_path_length, obs_dim, action_dim, device):
    num_rollouts = start_states.shape[0]
    observations = torch.zeros((num_rollouts, max_path_length+1, obs_dim), device=device)
    observations[:, 0] = torch.from_numpy(start_states).float().to(device)
    actions = torch.zeros((num_rollouts, max_path_length, action_dim), device=device)
    rewards = torch.zeros((num_rollouts, max_path_length, 1), device=device)
    terminals = torch.zeros((num_rollouts, max_path_length, 1), device=device)
    return observations, actions, rewards, terminals


def _sample_from_model(dynamics_model, state_actions):
    return dynamics_model.sample(state_actions)


def _get_prediction(sample_from_model, dynamics_model, states, actions, terminal_cutoff):
    state_actions = torch.cat([states, actions], dim=-1)
    transitions = sample_from_model(dynamics_model, state_actions)
    if (transitions != transitions).any():
        print('WARNING: NaN TRANSITIONS IN DYNAMICS MODEL ROLLOUT')
        transitions[transitions != transitions] = 0

    rewards = transitions[:, :1]
    dones = (transitions[:, 1:2] > terminal_cutoff).float()
    delta_obs = transitions[:, 2:]

    return rewards, dones, delta_obs


def _create_paths(observations, actions, rewards, terminals, max_path_length):
    observations_np = get_numpy(observations)
    actions_np = get_numpy(actions)
    rewards_np = get_numpy(rewards)
    terminals_np = get_numpy(terminals)

    paths = []
    for i in range(len(observations)):
        rollout_len = 1
        while rollout_len < max_path_length and terminals[i, rollout_len-1, 0] < 0.5:  # just check 0 or 1
            rollout_len += 1
        paths.append(dict(
            observations=observations_np[i, :rollout_len],
            actions=actions_np[i, :rollout_len],
            rewards=rewards_np[i, :rollout_len],
            next_observations=observations_np[i, 1:rollout_len + 1],
            terminals=terminals_np[i, :rollout_len],
        ))
    return paths


def _get_policy_actions(states, action_kwargs):
    policy = action_kwargs['policy']
    actions, *_ = policy.forward(states)
    return actions


def _model_rollout(
        dynamics_model,                             # torch dynamics model: (s, a) --> (r, d, s')
        start_states,                               # numpy array of states: (num_rollouts, obs_dim)
        get_action,                                 # method for getting action
        device,
        action_kwargs=None,                         # kwargs for get_action (ex. policy or actions)
        max_path_length=1000,                       # maximum rollout length (if not terminated)
        terminal_cutoff=None,                       # output Done if model pred > terminal_cutoff
        create_full_tensors=_create_full_tensors,
        sample_from_model=_sample_from_model,
        get_prediction=_get_prediction,
        create_paths=_create_paths,
):
    if action_kwargs is None:
        action_kwargs = dict()
    if terminal_cutoff is None:
        terminal_cutoff = float('inf')
    if max_path_length is None:
        raise ValueError('Must specify max_path_length in rollout function')

    obs_dim = dynamics_model.obs_dim
    action_dim = dynamics_model.action_dim

    s, a, r, d = create_full_tensors(start_states, max_path_length, obs_dim, action_dim, device=device)
    for t in range(max_path_length):
        a[:, t] = get_action(s[:, t], action_kwargs)
        r[:, t], d[:, t], delta_t = get_prediction(
            sample_from_model,
            dynamics_model,
            s[:, t], a[:, t],
            terminal_cutoff=terminal_cutoff,
        )
        s[:, t+1] = s[:, t] + delta_t

    paths = create_paths(s, a, r, d, max_path_length)

    return paths


def policy(dynamics_model, policy, start_states, device, max_path_length=1000, terminal_cutoff=None, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_actions,
        device=device,
        action_kwargs=dict(policy=policy),
        max_path_length=max_path_length,
        terminal_cutoff=terminal_cutoff,
        **kwargs,
    )
