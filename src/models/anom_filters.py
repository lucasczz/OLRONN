import torch


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


class AutoencoderBasedWeighter:
    def __init__(self, decay=0.99, weight_by="threshold_proximity", sharpness=1.0):
        self.decay = decay
        self.weight_by = weight_by
        self.sharpness = sharpness  # Hyperparameter for the Gaussian "hump"
        self.eps = 1e-8

        self.ema_value = 0.0

        self.ema_error = 0.0

        self.t = 1

    def get_weight(self, error: torch.Tensor) -> torch.Tensor:
        if self.weight_by == "easiness":
            value = -error

        elif self.weight_by == "difficulty":
            value = torch.log1p(error)

        elif self.weight_by == "threshold_proximity":
            ema_error_corrected = (
                self.ema_error / (1 - self.decay ** (self.t - 1)) if self.t > 1 else 0.0
            )

            value = torch.exp(
                -((error - ema_error_corrected) ** 2) / (2 * self.sharpness**2)
            )

        # --- Step 2: Update the EMAs with the current values ---

        # Update the EMA of the raw value
        self.ema_value = self.decay * self.ema_value + (1 - self.decay) * value.item()

        # Update the EMA of the reconstruction error
        self.ema_error = self.decay * self.ema_error + (1 - self.decay) * error

        # --- Step 3: Normalize the weight and return ---

        # Bias-correct the value EMA
        ema_value_corrected = self.ema_value / (1 - self.decay**self.t)
        self.t += 1

        # Scale the current value by the running average to stabilize the weights
        final_weight = value / (ema_value_corrected + self.eps)

        return final_weight


class EMAThresholder:
    def __init__(
        self,
    ):
        pass


class ConstantThresholder:
    pass
