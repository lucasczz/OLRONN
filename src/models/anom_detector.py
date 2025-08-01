import torch
import math
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from river import stats, utils
from scipy.special import ndtr


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        in_features,
        dropout=0.1,
        n_hidden=50,
        stat_window_size=250,
        lr=1e-3,
        p0=None,
    ):
        super().__init__()
        self.lr = lr
        self.fc_in = nn.Linear(in_features, n_hidden)
        self.fc_out = nn.Linear(n_hidden, in_features)
        self.dropout = nn.Dropout(dropout)
        self.var = utils.Rolling(stats.Var(), window_size=stat_window_size)
        self.mean = utils.Rolling(stats.Mean(), window_size=stat_window_size)
        self.optimizer = None
        self.p0 = p0

    def forward(self, x):
        x = self.dropout(x)
        h = F.selu(self.fc_in(x))
        return F.sigmoid(self.fc_out(h))

    def get_outlier_proba(self, x):
        if self.optimizer is None:
            self.optimizer = self.configure_optimizers()

        self.eval()
        with torch.inference_mode():
            x_pred = self(x)
            loss = F.smooth_l1_loss(x_pred, x)

        loss_item = loss.numpy(force=True)
        self.mean.update(loss_item)
        self.var.update(loss_item)
        loss_scaled = (loss_item - self.mean.get()) / math.sqrt(self.var.get())
        p = ndtr(loss_scaled)

        self.train()
        x_pred = self(x)
        loss = F.smooth_l1_loss(x_pred, x)
        if self.p0 is not None:
            loss = (1 - p / self.p0) * loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return p

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)
