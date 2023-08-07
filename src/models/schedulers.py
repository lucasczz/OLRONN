import inspect
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.optimizer import Optimizer

from river.drift import ADWIN

from src.models.drift_detectors import OneTailedPKSWIN


class DriftProbaLR(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        lr,
        last_epoch=-1,
    ) -> None:
        self._step_count = 0
        self.last_epoch = last_epoch
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.drift_detector = OneTailedPKSWIN()
        self.lr = lr
        self.optimizer = optimizer
        self.drift_detected = False

    def step(self, metrics, epoch=None):
        self._step_count += 1
        p_drift = self.drift_detector.update(metrics)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] += (base_lr - group["lr"]) * p_drift * self.lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class DriftResetLR(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        drift_detector_fn=ADWIN,
        detection_confidence=0.001,
        reset_weights=False,
        last_epoch=-1,
    ) -> None:
        self._step_count = 0
        self.last_epoch = last_epoch
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.drift_detector = (
            drift_detector_fn(detection_confidence)
            if detection_confidence > 0
            else None
        )
        self.reset_weights = reset_weights
        self.resettable_states = ["exp_avg", "exp_avg_sq", "momentum", "step"]

        self.optimizer = optimizer
        self.drift_detected = False

    def step(self, metrics, epoch=None):
        self._step_count += 1
        if self.drift_detector is not None:
            drift_detected = self.drift_detector.update(metrics).drift_detected
            if drift_detected:
                self.reset_states()
                if self.reset_weights:
                    self._reset_weights()

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def reset_states(self):
        if hasattr(self.optimizer, "_first_step"):
            self.optimizer._first_step = True

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr
            for p in group["params"]:
                state = self.optimizer.state[p]
                for resettable_state in self.resettable_states:
                    if resettable_state in state:
                        state[resettable_state] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

    def _reset_weights(self):
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.dim() < 2:
                    torch.nn.init.normal_(param)
                else:
                    torch.nn.init.xavier_uniform_(param)


class ManualLR(lr_scheduler.LRScheduler):
    def __init__(self, optimizer: Optimizer, gammas: dict) -> None:
        self.gammas = gammas
        super().__init__(optimizer)
        self._step_count = 0

    def get_lr(self):
        if self._step_count in self.gammas:
            return [
                group["lr"] * self.gammas[self._step_count]
                for group in self.optimizer.param_groups
            ]
        else:
            return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        self._step_count += 1
        values = self.get_lr()
        for value, group in zip(values, self.optimizer.param_groups):
            group["lr"] = value

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class LRLimiter(lr_scheduler.LRScheduler):
    def __init__(self, optimizer: Optimizer, min_lr=0.1, max_lr=None):
        for group in optimizer.param_groups:
            _max_lr = max_lr * group["lr"] if max_lr is not None else None
            _min_lr = min_lr * group["lr"] if min_lr is not None else None
            group.setdefault("max_lr", _max_lr)
            group.setdefault("min_lr", _min_lr)
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return [
            np.clip(group["lr"], group["min_lr"], group["max_lr"])
            for group in self.optimizer.param_groups
        ]


class ChainedScheduler(lr_scheduler.LRScheduler):
    def __init__(self, schedulers) -> None:
        self._schedulers = schedulers
        self.uses_metric = [
            "metrics" in inspect.signature(scheduler.step).parameters
            for scheduler in self._schedulers
        ]
        self.optimizer = schedulers[0].optimizer
        self._last_lr = [
            group["lr"] for group in self._schedulers[-1].optimizer.param_groups
        ]

    def step(self, metrics=None, epoch=None):
        for uses_metric, scheduler in zip(self.uses_metric, self._schedulers):
            if uses_metric:
                scheduler.step(metrics=metrics, epoch=epoch)
            else:
                scheduler.step(epoch=epoch)
        self._last_lr = [
            group["lr"] for group in self._schedulers[-1].optimizer.param_groups
        ]


def get_cyclic_scheduler(optimizer, max_lr, step_size_up=4000):
    base_lr = optimizer.param_groups[0]["lr"]
    return CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        cycle_momentum=False,
    )
