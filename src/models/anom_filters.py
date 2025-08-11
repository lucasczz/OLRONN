import torch
import numpy as np
import math
import torch.nn.functional as F


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
        model,
        threshold_quantile,
        threshold_type="sigmoid",
        steepness=20,
        device="cpu",
    ):
        self.model = model.to(device)
        self.threshold_type = threshold_type
        self.steepness = steepness
        self.device = device
        self.model.eval()
        self.threshold_quantile = threshold_quantile

        self.threshold = None
        self.weight_correction = None
        self.adjusted_steepness = None

    def __call__(self, x: torch.Tensor):
        x_rec = self.model(x)
        anom_score = F.l1_loss(x, x_rec).detach().cpu().item()

        if self.threshold_type == "hard":
            return int(anom_score <= self.threshold) * self.weight_correction
        elif self.threshold_type == "linear":
            return (1 - anom_score / self.threshold) * self.weight_correction
        elif self.threshold_type == "sigmoid":
            return self.weight_correction / (
                1 + math.exp(self.adj_steepness * (anom_score - self.threshold))
            )

    def calibrate(self, x_val):
        with torch.inference_mode():
            x_val = x_val.to(self.device)
            x_rec = self.model(x_val)
            anom_scores = F.l1_loss(x_val, x_rec, reduction="none").mean(dim=-1)

        anom_scores = anom_scores.numpy(force=True)
        self.threshold = (np.quantile(anom_scores, self.threshold_quantile)).item()
        self.adj_steepness = (self.steepness / np.quantile(anom_scores, 0.95)).item()

        if self.threshold_type == "hard":
            anom_weights = (anom_scores <= self.threshold).astype(int)
        elif self.threshold_type == "linear":
            anom_weights = -anom_scores / self.threshold
        else:
            anom_weights = 1 / (
                1 + np.exp(self.adj_steepness * (anom_scores - self.threshold))
            )
        self.weight_correction = (1 / anom_weights.mean()).item()
