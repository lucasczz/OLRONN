import csv
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset, RandomSampler
import inspect
from itertools import product
from dog import DoG, PolynomialDecayAverager
import zipfile
import os
from typing import Iterable, List
import warnings
import torch.multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from src.data.datasets import get_dataset
from src.models.networks import get_mlp
import os
from src.models.schedulers import KunchevaLR, LRLimiter

from src.utils import ExperimentLogger, one_hot

REPORTS_PATH = Path(__file__).parent.parent.joinpath("reports")

N_PROCESSES = 4
DEVICE = "cpu"

BATCH_SIZE = 4
N_HIDDEN_LAYERS = 1
SEEDS = list(range(5))
LRS = [2**-i for i in range(-1, 11)]

DATASET_NAMES = [
    "RBF abrupt",
    "RBF incr.",
    "Insects gradual",
    "Insects abrupt",
    "Insects incr.",
    "Electricity",
    "Covertype",
]


def bce_with_logits(y, logits, weight=None):
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = torch.mean(loss, dim=list(range(1, loss.dim())))
    if weight is not None:
        loss = loss * weight
    return loss


def accuracy(y_pred, y_true):
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(y_pred == y_true)


def write_result(result, log_path):
    has_header = False
    if log_path.exists():
        with open(log_path, "r") as f:
            has_header = f.read(1024) != ""
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        writer = csv.DictWriter(f, result.keys())
        if not has_header:
            writer.writeheader()
        writer.writerow(result)


def tune_prequential(
    base_lr,
    log_path,
    gamma,
    seed,
    data,
    n_samples=1000,
    batch_size=4,
    n_hidden_layers=1,
    n_hidden_units=None,
    steps=10000,
    verbose=False,
    **log_info,
):
    torch.manual_seed(seed)
    x_sample, y_sample = data[0]
    net = get_mlp(
        x_sample.shape[-1],
        y_sample.shape[-1],
        n_hidden_layers=n_hidden_layers,
        hidden_features=n_hidden_units,
    )
    optim = torch.optim.SGD(net.parameters(), lr=base_lr)
    if gamma < 1:
        scheduler = ExponentialLR(optim, gamma=gamma)
        lr_limiter = LRLimiter(optim, min_lr=0.1)
    data = Subset(data, range(n_samples))
    sampler = RandomSampler(data, num_samples=steps, replacement=True)
    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    logger = ExperimentLogger(
        optim,
        metric_fns=[accuracy],
        log_path=log_path,
        logged_variables=["lr"],
        parameters={
            "base_lr": base_lr,
            "batch_size": batch_size,
            "seed": seed,
            "n_hidden_layers": n_hidden_layers,
            "n_hidden_units": n_hidden_units,
            "n_samples": n_samples,
            "gamma": gamma,
            "schedule": "Exponential",
            **log_info,
        },
    )
    iterator = tqdm(dataloader) if verbose else dataloader
    for x, y in iterator:
        # Perform forward pass
        logits = net(x)
        loss = bce_with_logits(y, logits)
        loss_sum = loss.sum()

        y_pred = torch.argmax(logits, dim=-1)
        y = torch.argmax(y, dim=-1)

        # Perform backward pass
        optim.zero_grad()
        loss_sum.backward()
        optim.step()
        if gamma < 1:
            scheduler.step()
            lr_limiter.step()
        logger.update(y=y, y_pred=y_pred, loss=loss)


def tune_batch_mode(
    base_lr,
    log_path,
    gamma,
    seed,
    data,
    batch_size=4,
    n_hidden_layers=1,
    n_hidden_units=None,
    n_samples=1000,
    validation_split=0.2,
    num_epochs=20,
    device="cpu",
    verbose=False,
    **log_info,
):
    torch.manual_seed(seed)
    x_sample, y_sample = data[0]
    net = get_mlp(
        x_sample.shape[-1],
        y_sample.shape[-1],
        n_hidden_layers=n_hidden_layers,
        hidden_features=n_hidden_units,
    )
    optim = torch.optim.SGD(net.parameters(), lr=base_lr)
    if gamma < 1:
        scheduler = ExponentialLR(optim, gamma=gamma)
        lr_limiter = LRLimiter(optim, min_lr=0.1)
    train_loader, val_loader = get_train_val_loaders(
        batch_size, n_samples, seed, data, validation_split
    )
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        if verbose:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            optim.zero_grad()
            logits = net(x_train)
            loss = bce_with_logits(y_train, logits).sum()
            loss.backward()
            optim.step()
            if gamma < 1:
                scheduler.step()
                lr_limiter.step()

            total_loss += loss.item()

        # Validation
        net.eval()
        with torch.inference_mode():
            total_val_loss = 0.0
            total_correct = 0

            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)

                val_logits = net(x_val)
                val_loss = bce_with_logits(y_val, val_logits).sum()
                total_correct += (
                    (torch.argmax(val_logits, dim=-1) == torch.argmax(y_val, dim=-1))
                    .sum()
                    .item()
                )
                total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_loader)
            best_val_loss = min(best_val_loss, average_val_loss)
            average_val_accuracy = total_correct / len(val_loader)

            if average_val_accuracy > best_val_accuracy:
                best_val_accuracy = average_val_accuracy
                best_epoch = epoch

            if verbose:
                print(f"Train loss: {total_loss / len(train_loader)}")
                print(f"Validation loss: {average_val_loss}")
                print(f"Validation accuracy: {average_val_accuracy}")

        result = {
            "base_lr": base_lr,
            "batch_size": batch_size,
            "seed": seed,
            "n_hidden_layers": n_hidden_layers,
            "n_hidden_units": n_hidden_units,
            "n_samples": n_samples,
            "gamma": gamma,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            **log_info,
        }
    write_result(result, log_path=log_path)
    return best_val_accuracy


def get_train_val_loaders(batch_size, n_samples, seed, data, validation_split):
    idcs = np.arange(n_samples)
    train_idcs, val_idcs = train_test_split(
        idcs, random_state=seed, test_size=validation_split
    )
    data_train = Subset(data, train_idcs)
    data_val = Subset(data, val_idcs)
    train_loader = DataLoader(data_train, batch_size=batch_size)
    val_loader = DataLoader(data_val, batch_size=batch_size)
    return train_loader, val_loader


def run_prequential(
    data: torch.Tensor,
    log_path: str,
    batch_size: int = 4,
    optim_fn: callable = torch.optim.SGD,
    scheduler_fn: callable = None,
    base_lr: float = None,
    metrics: List[callable] = [accuracy],
    net_params: dict = {},
    log_variables: List[str] = [],
    log_lr_norms: bool = False,
    net_fn: callable = get_mlp,
    log_interval: int = 100,
    seed: int = 42,
    verbose: bool = False,
    device: str = "cpu",
    **log_info,
):
    torch.manual_seed(seed)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    sample_x, sample_y = data[0]
    # Initialize model
    parameters = {
        "base_lr": base_lr,
        "seed": seed,
        "batch_size": batch_size,
        **log_info,
    }
    net = net_fn(
        in_features=sample_x.shape[-1], out_features=sample_y.shape[-1], **net_params
    ).to(device)
    # Initialize optimizer and tracking of optimization metrics
    optim, averager, scheduler, sched_uses_metric, lr_limiter = init_optim(
        optim_fn, scheduler_fn, base_lr, net
    )

    exp_tracker = ExperimentLogger(
        optim=optim,
        parameters=parameters,
        logged_variables=log_variables,
        metric_fns=metrics,
        log_path=log_path,
        log_interval=log_interval,
        log_lr_norms=log_lr_norms,
    )

    iterator = tqdm(dataloader) if verbose else dataloader
    for x, y in iterator:
        # Perform forward pass
        logits = net(x)
        loss = bce_with_logits(y, logits)
        loss_sum = loss.sum()

        y_pred = torch.argmax(logits, dim=-1)
        y = torch.argmax(y, dim=-1)

        # Perform backward pass
        optim.zero_grad()
        loss_sum.backward()
        loss_sum = loss_sum.cpu().item()
        optim.step()
        if averager is not None:
            averager.step()
        exp_tracker.update(y=y, y_pred=y_pred, loss=loss)

        if scheduler_fn is not None:
            if sched_uses_metric:
                scheduler.step(loss_sum)
            else:
                scheduler.step()
        if lr_limiter is not None:
            lr_limiter.step()


def init_optim(optim_fn, scheduler_fn, base_lr, net):
    scheduler = None
    lr_limiter = None
    sched_uses_metric = False
    optim = (
        optim_fn(net.parameters(), lr=base_lr)
        if base_lr is not None
        else optim_fn(net.parameters())
    )
    averager = PolynomialDecayAverager(net) if isinstance(optim, DoG) else None
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optim)
        sched_uses_metric = "metrics" in inspect.signature(scheduler.step).parameters
        if not isinstance(scheduler, KunchevaLR):
            lr_limiter = LRLimiter(optim)

    return optim, averager, scheduler, sched_uses_metric, lr_limiter


def get_config_grid(search_space=[{}], fixed_kwargs={}, **kwargs):
    results = []
    for row in search_space:
        row.update(kwargs)
        fixed_attributes = fixed_kwargs.copy()
        variable_keys = []
        variable_values = []
        for key, value in row.items():
            if isinstance(value, (tuple, list)):
                variable_values.append(value)
                variable_keys.append(key)
            else:
                fixed_attributes[key] = value
        configs = [
            fixed_attributes | dict(zip(variable_keys, prod_values))
            for prod_values in product(*variable_values)
        ]
        results.extend(configs)
    return results


def run_configs(
    dataset_names: List[str],
    configs: Iterable[Iterable],
    log_path: str,
    run_fn=run_prequential,
    debug: bool = False,
    dtype: torch.dtype = torch.float32,
    subsample: int = None,
    n_processes: int = N_PROCESSES,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for dataset_name in dataset_names:
        # Data preparation
        x, y = get_dataset(dataset_name)
        if subsample is not None:
            x = x[subsample:]
            y = y[subsample:]
        y = one_hot(y)
        x_t = torch.tensor(x, dtype=dtype, device=DEVICE).share_memory_()
        y_t = torch.tensor(y, dtype=dtype, device=DEVICE).share_memory_()
        data = TensorDataset(x_t, y_t)

        if debug == True:
            warnings.warn("Running in debug mode!")
            for config in configs[:5]:
                run_fn(
                    data=data,
                    dataset=dataset_name,
                    verbose=True,
                    log_path=log_path,
                    **config,
                )
            return
        else:
            pbar = tqdm(total=len(configs), desc=dataset_name)
            with mp.get_context("spawn").Pool(processes=n_processes) as pool:
                for config in configs:
                    config["data"] = data
                    config["dataset"] = dataset_name
                    config["log_path"] = log_path
                    pool.apply_async(
                        func=run_fn,
                        kwds=config,
                        callback=lambda x: pbar.update(),
                        error_callback=print,
                    )

                pool.close()
                pool.join()


def zip_csv(csv):
    with zipfile.ZipFile(f"{csv}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(csv)
    os.remove(csv)
