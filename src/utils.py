from pathlib import Path
import time
from typing import Callable, Dict, List
import numpy as np
import torch
import csv


def one_hot(arr):
    # Get the unique values in the array
    unique_values = np.unique(arr)

    # Perform one-hot encoding
    return np.eye(len(unique_values))[np.searchsorted(unique_values, arr)]


class ExperimentLogger:
    def __init__(
        self,
        optim: torch.optim.Optimizer,
        metric_fns: List[Callable],
        parameters: Dict,
        log_path: str,
        logged_variables: List[str] = [],
        log_lr_norms: bool = False,
        log_interval: int = 100,
    ) -> None:
        self.optim = optim
        self.logged_vars = (
            logged_variables
            if isinstance(logged_variables, (list, tuple))
            else [logged_variables]
        )
        self.metric_fns = metric_fns
        self.parameters = parameters
        self.save_path = Path(log_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.tmp_variables = {var_name: [] for var_name in self.logged_vars}
        self.tmp_variables["loss"] = []
        self.history = []
        self.params = []
        # Get all parameters
        for group in self.optim.param_groups:
            for param in group["params"]:
                self.params.append(param)
        self.step = 0
        self.labels = []
        self.preds = []
        self.last_log_step = 0
        self.log_lr_norms = log_lr_norms
        if log_lr_norms:
            self.tmp_variables.update({"lr_norm": []})
            self.last_param_state = torch.cat(
                [param.view(-1) for param in self.optim.param_groups[0]["params"]]
            ).numpy(force=True)
        self.last_time = time.time()

    def update(self, y: torch.Tensor, y_pred: torch.Tensor, loss: torch.Tensor):
        if self.step + y.shape[0] > self.last_log_step + self.log_interval:
            cutoff = (self.step + y.shape[0]) % self.log_interval
            args0 = y[:-cutoff], y_pred[:-cutoff], loss[:-cutoff]
            args1 = y[-cutoff:], y_pred[-cutoff:], loss[-cutoff:]
            self._update(*args0)
            self._update(*args1)
        else:
            self._update(y, y_pred, loss)

    def _update(self, y, y_pred, loss):
        self.step += y.shape[0]
        # Log loss
        self.tmp_variables["loss"].append(loss.mean().cpu().item())

        # Log predictions and labels
        self.labels.append(y.numpy(force=True))
        self.preds.append(y_pred.numpy(force=True))
        if self.log_lr_norms:
            # Log norm of gradient
            grad = torch.cat([param.grad.view(-1) for param in self.params])
            grad = grad.numpy(force=True)
            # Log step size
            current_param_state = torch.cat(
                [param.view(-1) for param in self.params]
            ).numpy(force=True)
            update = current_param_state - self.last_param_state
            mask = grad != 0
            lrs = np.divide(update, grad, where=mask, out=np.zeros_like(grad))
            if (~mask).any():
                lrs[~mask] = lrs[mask].mean()
            lr_norm = np.linalg.norm(lrs)
            self.tmp_variables["lr_norm"].append(lr_norm / np.sqrt(lrs.size))

            self.last_param_state = current_param_state

        # Get values of variables to be logged from optimizer
        for var_name in self.logged_vars:
            tmp = []
            found_in_group = False
            for group in self.optim.param_groups:
                if var_name in group:
                    tmp.append(group[var_name])
                    found_in_group = True
            if not found_in_group:
                for param in self.params:
                    state = self.optim.state[param]
                    if var_name in state:
                        tmp.append(state[var_name].view(-1))
            if len(tmp) > 0:
                if isinstance(tmp[0], torch.Tensor):
                    tmp = torch.cat(tmp).cpu().detach().numpy()
                self.tmp_variables[var_name].append(np.mean(tmp, axis=0))
        # Calculate metrics and save average of values tracked since last logging step
        if self.step % self.log_interval == 0:
            self.log_step()

    def log_step(self):
        results = {k: v for k, v in self.parameters.items()}
        results["step"] = self.step
        results["runtime"] = time.time() - self.last_time
        self.preds = np.concatenate(self.preds)
        self.labels = np.concatenate(self.labels)

        # Calculate metrics based on predictions and labels captured since last loggin step
        for metric in self.metric_fns:
            results[metric.__name__] = metric(self.preds, self.labels)
        self.labels = []
        self.preds = []

        # Calculate mean of variables
        for var_name, values in self.tmp_variables.items():
            if len(values) == 0:
                values = [np.NAN]
            results[var_name] = np.mean(np.stack(values), axis=0)
            if isinstance(results[var_name], np.ndarray):
                results[var_name] = results[var_name].tobytes()
            self.tmp_variables[var_name] = []

        self._write_step(results)
        self.last_log_step = self.step
        self.last_time = time.time()

    def _write_step(self, results: Dict):
        # Check if file to write to has a .csv header already
        has_header = False
        if self.save_path.exists():
            with open(self.save_path, "r") as f:
                has_header = f.read(1024) != ""

        # Write results captured since last logging step
        with open(self.save_path, "a") as f:
            writer = csv.DictWriter(f, results.keys())
            if not has_header:
                writer.writeheader()
            writer.writerow(results)
