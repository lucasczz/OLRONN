import torch
import numpy as np
import math
import torch.nn.functional as F
from src.models.networks import get_autoencoder
from tqdm import tqdm
from river import stats, utils


class LogitBasedWeighter:
    def __init__(self, decay=0.99, weight_by="threshold_proximity"):
        self.decay = decay
        self.ema = 0.0
        self.t = 1
        self.weight_by = weight_by
        self.eps = 1e-6

    def get_weight(self, logits, label):
        if self.weight_by == "threshold_proximity":
            p_right = torch.softmax(logits, dim=-1)[:, label]
            value = p_right * (1 - p_right)
        elif self.weight_by == "easiness":
            value = torch.softmax(logits, dim=-1)[:, label]
        else:
            value = 1 - torch.softmax(logits, dim=-1)[:, label]

        self.ema = self.decay * self.ema + (1 - self.decay) * value
        ema_corrected = self.ema / (1 - self.decay**self.t)
        return value / (ema_corrected + self.eps)


class AEFilter:
    def __init__(
        self,
        n_hidden_layers,
        n_hidden_units,
        lr_pretraining,
        lr_online,
        threshold_quantile,
        window_size=1000,
        dropout=0,
        epochs=5,
        threshold_type="sigmoid",
        steepness=20,
        device="cpu",
    ):
        self.model = None
        self.optimizer = None
        self.threshold_type = threshold_type

        self.steepness = steepness
        self.device = device
        self.threshold_quantile = threshold_quantile

        self.threshold = None
        self.adjusted_steepness = None

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.lr_pretraining = lr_pretraining
        self.lr_online = lr_online
        self.dropout = dropout
        self.epochs = epochs

        self.rolling_q = stats.RollingQuantile(
            q=threshold_quantile, window_size=window_size
        )
        self.rolling_mean = utils.Rolling(stats.Mean(), window_size=window_size)

    def __call__(self, x: torch.Tensor):
        if self.model is None:
            self.model = get_autoencoder(
                in_features=x.shape[-1],
                n_hidden_units=self.n_hidden_units,
                n_hidden_layers=self.n_hidden_layers,
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        x_rec = self.model(x)
        anom_score = F.l1_loss(x, x_rec).detach().cpu().item()
        sample_weight = self._get_sample_weight(anom_score)

        if self.lr_online > 0:
            self.tuning_step(x)

        rolling_mean = self.rolling_mean.get()
        if rolling_mean > 0:
            scaled_sample_weight = sample_weight / rolling_mean
        else:
            scaled_sample_weight = sample_weight

        self.rolling_q.update(anom_score)
        self.rolling_mean.update(sample_weight)
        return scaled_sample_weight

    def _get_sample_weight(self, anom_score):
        threshold = self.rolling_q.get()
        if self.threshold_type == "hard":
            if isinstance(anom_score, np.ndarray):
                weight = (anom_score <= threshold).astype(int)
            else:
                weight = int(anom_score <= threshold)
        elif self.threshold_type == "linear":
            weight = 1 - anom_score / (2 * threshold)
        elif self.threshold_type == "sigmoid":
            weight = 1 / (
                1 + np.exp(self.steepness / threshold * (anom_score - threshold))
            )

        return weight

    def pre_train(self, x_pre, verbose=False):
        self.model = pretrain_autoencoder(
            x_pre,
            self.n_hidden_layers,
            self.n_hidden_units,
            lr=self.lr_pretraining,
            epochs=self.epochs,
            device=self.device,
            dropout=self.dropout,
        )

    def tuning_step(self, x_tune):
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr_online)

        self.model.train()
        x_rec = self.model(x_tune)
        loss = F.l1_loss(x_tune, x_rec)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calibrate(self, x_val, is_anom=None):
        with torch.inference_mode():
            x_val = x_val.to(self.device)
            x_rec = self.model(x_val)
            anom_scores = F.l1_loss(x_val, x_rec, reduction="none").mean(dim=-1)

        anom_scores = anom_scores.numpy(force=True)

        for anom_score in anom_scores:
            self.rolling_q.update(anom_score)

        anom_weights = self._get_sample_weight(anom_scores)

        for anom_weight in anom_weights:
            self.rolling_mean.update(anom_weight)


def pretrain_autoencoder(
    x_pre,
    n_hidden_layers,
    n_hidden_units,
    lr,
    epochs,
    device,
    dropout=0,
    model=None,
    verbose=False,
):
    optimizer = None

    for i in range(epochs):
        iterator = tqdm(x_pre) if verbose else x_pre
        for x in iterator:
            if model is None:
                model = get_autoencoder(
                    in_features=x.shape[-1],
                    dropout=dropout,
                    n_hidden_units=n_hidden_units,
                    n_hidden_layers=n_hidden_layers,
                )
                model = model.to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            x = x.to(device)

            model.train()
            x_rec = model(x)
            loss = F.l1_loss(x_rec, x)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model
