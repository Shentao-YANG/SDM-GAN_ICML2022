import abc
from collections import OrderedDict
from utils.utils import np_to_pytorch_batch, get_numpy, print_banner
from utils.logger import create_stats_ordered_dict, logger
import numpy as np
import torch
import torch.optim as optim


class TorchTrainer(object, metaclass=abc.ABCMeta):
    def __init__(self, device):
        self._num_train_steps = 0
        self.device = device

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch, device=self.device)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self):
        pass


class ModelTrainer(TorchTrainer):
    def __init__(
            self,
            ensemble,
            device,
            replay_buffer,
            num_elites=None,
            learning_rate=1e-3,
            batch_size=256,
            optimizer_class=optim.Adam,
    ):
        super().__init__(device=device)

        self.ensemble = ensemble
        self.ensemble_size = ensemble.ensemble_size
        self.num_elites = min(num_elites, self.ensemble_size) if num_elites else self.ensemble_size

        self.obs_dim = ensemble.obs_dim
        self.action_dim = ensemble.action_dim
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self._state = {}
        self._snapshots = {i: (None, 1e32) for i in range(self.ensemble_size)}

        self.optimizer = self.construct_optimizer(optimizer_class, learning_rate)

    def construct_optimizer(self, optimizer_class, lr):
        decays = [.000025, .00005, .000075, .000075, .0001]
        fcs = self.ensemble.fcs + [self.ensemble.last_fc]

        if self.ensemble.separate_mean_var:
            decays.append(.0001)
            fcs += [self.ensemble.last_fc_std]

        opt_params = [{'params': fcs[i].parameters(), 'weight_decay': decays[i]} for i in range(len(fcs))]
        opt_params.extend([{'params': self.ensemble.max_logstd, 'weight_decay': 0.}, {'params': self.ensemble.min_logstd, 'weight_decay': 0.}])

        optimizer = optimizer_class(opt_params, lr=lr)

        return optimizer

    def save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / abs(best)
            if improvement > 0.01:
                self._snapshots[i] = (epoch, np.mean(get_numpy(current)))
                self._state[i] = self.ensemble.get_idv_model_state(i)
                updated = True

        return updated

    def train_from_buffer(self, holdout_pct=0.2, max_grad_steps=1000, epochs_since_last_update=5):

        data = self.replay_buffer.get_transitions()
        x = data[:, :self.obs_dim + self.action_dim]    # inputs  s, a
        y = data[:, self.obs_dim + self.action_dim:]    # predict r, d, ns
        y[:, -self.obs_dim:] -= x[:, :self.obs_dim]     # predict delta in the state

        # get normalization statistics
        self.ensemble.fit_input_stats(data=x, y=y)

        # normalize the delta in y_test
        y[..., self.ensemble.rns_np] = (y[..., self.ensemble.rns_np] - self.ensemble.delta_obs_mu.data.cpu().numpy()) / self.ensemble.delta_obs_std.data.cpu().numpy()

        # generate holdout set
        inds = np.random.permutation(data.shape[0])
        x, y = x[inds], y[inds]

        n_train = max(int((1. - holdout_pct) * data.shape[0]), data.shape[0] - 8092)

        x_train, y_train = x[:n_train], y[:n_train]
        x_test, y_test = x[n_train:], y[n_train:]
        x_test, y_test = torch.from_numpy(x_test).float().to(self.device), torch.from_numpy(y_test).float().to(self.device)

        # train until holdout set converge
        num_epochs, num_steps = 0, 0
        num_epochs_since_last_update = 0
        best_holdout_loss = float('inf')
        num_batches = int(np.ceil(n_train / self.batch_size))

        while num_epochs_since_last_update < epochs_since_last_update and num_steps < max_grad_steps:
            # generate idx for each model to bootstrap
            self.ensemble.train()
            for b in range(num_batches):
                b_idxs = np.random.randint(n_train, size=(self.ensemble_size * self.batch_size))
                x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]
                x_batch, y_batch = torch.from_numpy(x_batch).float().to(self.device), torch.from_numpy(y_batch).float().to(self.device)
                x_batch = x_batch.view(self.ensemble_size, self.batch_size, -1)
                y_batch = y_batch.view(self.ensemble_size, self.batch_size, -1)
                loss = self.ensemble.get_loss(x_batch, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # stop training based on holdout loss improvement
            self.ensemble.eval()
            with torch.no_grad():
                holdout_errors, holdout_losses = self.ensemble.get_loss(x_test, y_test, split_by_model=True, return_l2_error=True)    # (losses, l2_errors)

            updated = self.save_best(epoch=num_epochs + 1, holdout_losses=holdout_losses)
            if num_epochs == 0 or updated:
                num_epochs_since_last_update = 0
            else:
                num_epochs_since_last_update += 1

            holdout_loss = sum(sorted(holdout_losses)[:self.num_elites]) / self.num_elites

            num_steps += num_batches
            num_epochs += 1

            if num_epochs % 1 == 0:
                logger.record_tabular('Model Training Epochs', num_epochs)
                logger.record_tabular('Num epochs since last update', num_epochs_since_last_update)
                logger.record_tabular('Model Training Steps', num_steps)
                logger.record_tabular("Model Training Loss", loss.cpu().data.numpy())
                logger.record_tabular("Model Holdout Loss", np.mean(get_numpy(sum(holdout_losses))) / self.ensemble_size)
                logger.record_tabular('Model Elites Holdout Loss', np.mean(get_numpy(holdout_loss)))

                for i in range(self.ensemble_size):
                    name = 'Model%d' % i
                    logger.record_tabular(name + ' Loss', np.mean(get_numpy(holdout_losses[i])))
                    logger.record_tabular(name + ' Error', np.mean(get_numpy(holdout_errors[i])))
                    logger.record_tabular(name + ' Best', self._snapshots[i][1])
                    logger.record_tabular(name + ' Best Epoch', self._snapshots[i][0])

                logger.dump_tabular(with_timestamp=False)

        self.ensemble.load_model_state_from_dict(state_dict=self._state)
        with torch.no_grad():
            holdout_errors, holdout_losses = self.ensemble.get_loss(x_test, y_test, split_by_model=True, return_l2_error=True)  # (losses, l2_errors)
        for i in range(self.ensemble_size):
            print_banner(f"Model {i}: Loss {np.mean(get_numpy(holdout_losses[i])):.8f}; Error {np.mean(get_numpy(holdout_errors[i])):.8f}; From epoch {self._snapshots[i][0]}")

        print_banner(f"OLD ensemble elites are {self.ensemble.elites}")
        self.ensemble.elites = np.argsort([np.mean(get_numpy(x)) for x in holdout_losses])[:self.num_elites]
        print_banner(f"NEW ensemble elites are {self.ensemble.elites}, their holdout losses are {[np.mean(get_numpy(holdout_losses[_idx])) for _idx in self.ensemble.elites]}")
        print_banner(f"MAX_logstd: {self.ensemble.max_logstd.data.cpu().numpy()}")
        print_banner(f"MIN_logstd: {self.ensemble.min_logstd.data.cpu().numpy()}")

        self._state = None

    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    @property
    def networks(self):
        return [
            self.ensemble
        ]

    def get_snapshot(self):
        return dict(
            ensemble=self.ensemble
        )
