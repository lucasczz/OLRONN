import inspect
from typing import Callable, Dict, List, Tuple
import torch
from torch.nn import init
import math
from collections import deque
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.optimizer import Optimizer

from src.models.drift_detectors import OneTailedADWIN


class KunchevaLR(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        window_size=32,
        last_epoch: int = ...,
        verbose: bool = ...,
    ) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.losses = deque([], maxlen=2 * window_size)
        self.window_size = window_size
        self._step_count = 0

    def step(self, loss, epoch=None):
        if isinstance(loss, torch.Tensor):
            loss = loss.numpy(force=True)
        self._step_count += 1
        self.losses.append(loss)
        if len(self.losses) == self.window_size * 2:
            loss_new = np.mean(list(self.losses)[self.window_size :])
            loss_old = np.mean(list(self.losses)[: self.window_size])
            delta_loss = loss_old - loss_new
            for group in self.optimizer.param_groups:
                group["lr"] = group["lr"] ** (1 + delta_loss)
                group["lr"] = min(group["lr"], 1.0)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class WeightResetLR(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        drift_detector_fn=OneTailedADWIN,
        detection_confidence=1e-4,
        last_epoch=-1,
    ) -> None:
        self._step_count = 0
        self.last_epoch = last_epoch
        self.drift_detector = drift_detector_fn(detection_confidence)
        self.optimizer = optimizer
        self.drift_detected = False
        self._init_optim_state = None

    def step(self, loss, epoch=None):
        if isinstance(loss, torch.Tensor):
            loss = loss.numpy(force=True)
        if self._step_count == 0:
            self._init_optim_state = self.optimizer.state_dict()
        self._step_count += 1
        if self.drift_detector is not None:
            drift_detected = self.drift_detector.update(loss).drift_detected
            if drift_detected:
                self.optimizer.load_state_dict(self._init_optim_state)
                self.reset_parameters()

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def reset_parameters(self) -> None:
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param.dim() > 1:
                    init.kaiming_uniform_(param, a=math.sqrt(5))
                    fan_in, _ = init._calculate_fan_in_and_fan_out(param)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                else:
                    init.uniform_(param, -bound, bound)


class DriftResetLR(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        drift_detector_fn=OneTailedADWIN,
        detection_confidence=1e-4,
        last_epoch=-1,
    ) -> None:
        self._step_count = 0
        self.last_epoch = last_epoch
        self.drift_detector = drift_detector_fn(detection_confidence)
        self.optimizer = optimizer
        self.drift_detected = False
        self._init_optim_state = None

    def step(self, loss, epoch=None):
        if isinstance(loss, torch.Tensor):
            loss = loss.numpy(force=True)
        if self._step_count == 0:
            self._init_optim_state = self.optimizer.state_dict()
        self._step_count += 1
        if self.drift_detector is not None:
            drift_detected = self.drift_detector.update(loss).drift_detected
            if drift_detected:
                self.optimizer.load_state_dict(self._init_optim_state)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


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
            group.setdefault("min_lr", _min_lr)
            group.setdefault("max_lr", _max_lr)
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return [
            np.clip(group["lr"], group["min_lr"], group["max_lr"])
            for group in self.optimizer.param_groups
        ]


def get_scheduler_chain(*schedulers: Callable):
    class SchedulerChain(lr_scheduler.LRScheduler):
        def __init__(
            self, optimizer: Optimizer, last_epoch: int = ..., verbose: bool = ...
        ) -> None:
            self.optimizer = optimizer
            self._schedulers = [scheduler(optimizer) for scheduler in schedulers]
            self._uses_metric = [
                "loss" in inspect.signature(scheduler.step).parameters
                for scheduler in self._schedulers
            ]
            self._last_lr = [
                group["lr"] for group in self._schedulers[-1].optimizer.param_groups
            ]

        def step(self, loss=None, epoch=None):
            if isinstance(loss, torch.Tensor):
                loss = loss.numpy(force=True)
            for uses_metric, scheduler in zip(self._uses_metric, self._schedulers):
                if uses_metric:
                    scheduler.step(loss=loss, epoch=epoch)
                else:
                    scheduler.step(epoch=epoch)
            self._last_lr = [
                group["lr"] for group in self._schedulers[-1].optimizer.param_groups
            ]

    return SchedulerChain


def get_cyclic_lr(optimizer, max_lr, step_size_up=4000):
    base_lr = optimizer.param_groups[0]["lr"]
    return CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size_up,
        cycle_momentum=False,
    )
