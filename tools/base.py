from functools import partial
import inspect
from dog import DoG, PolynomialDecayAverager
import itertools
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
from src.data.datasets import DATASETS, get_dataset
from src.models.drift_detectors import OneTailedPKSWIN
from src.models.networks import get_mlp
import os
from src.models.schedulers import DriftProbaLR

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
    "SEA",
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


def run_prequential(
    data: torch.Tensor,
    log_path: str,
    batch_size: int = 4,
    optim_fn: callable = torch.optim.SGD,
    scheduler_fn: callable = None,
    drift_detector: callable = None,
    reset_weights: bool = False,
    base_lr: float = None,
    metrics: List[callable] = [accuracy],
    net_params: dict = {},
    log_variables: List[str] = [],
    log_lr_norms: bool = False,
    net_fn: callable = get_mlp,
    log_interval: int = 100,
    seed: int = 42,
    verbose: bool = True,
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
    optim, averager, scheduler = init_optim(optim_fn, scheduler_fn, base_lr, net)

    sched_uses_metric = "metrics" in inspect.signature(scheduler.step).parameters
    exp_tracker = ExperimentLogger(
        optim=optim,
        parameters=parameters,
        logged_variables=log_variables,
        metric_fns=metrics,
        save_path=log_path,
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
        if isinstance(drift_detector, OneTailedPKSWIN): 
            p_drift = drift_detector.update(loss_sum)
            for group in optim.param_groups:
                group["lr"] += (base_lr - group["lr"]) * p_drift 
        elif drift_detector is not None:
            drift_detected = drift_detector.update(loss_sum).drift_detected
            if drift_detected:
                if reset_weights:
                    net = net_fn(
                        in_features=sample_x.shape[-1],
                        out_features=sample_y.shape[-1],
                        **net_params,
                    ).to(device)
                optim, averager, scheduler = init_optim(
                    optim_fn, scheduler_fn, base_lr, net
                )
                exp_tracker.optim = optim


def init_optim(optim_fn, scheduler_fn, base_lr, net):
    if base_lr is not None:
        optim = optim_fn(net.parameters(), lr=base_lr)
    else:
        optim = optim_fn(net.parameters())
    if isinstance(optim, DoG):
        averager = PolynomialDecayAverager(net)
    else:
        averager = None
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optim)
    else:
        scheduler = None

    return optim, averager, scheduler


def print_error(error):
    print(error)


def test_configs(
    test_func: callable,
    dataset_names: List[str],
    configs: Iterable[Iterable],
    debug: bool = False,
    dtype: torch.dtype = torch.float32,
    subsample=None,
    n_processes=N_PROCESSES,
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
                test_func(data=data, dataset_name=dataset_name, *config, verbose=True)
            return
        else:
            pbar = tqdm(total=len(configs), desc=dataset_name)
            with mp.get_context("spawn").Pool(processes=n_processes) as pool:
                for config in configs:
                    pool.apply_async(
                        func=partial(test_func, data=data, dataset_name=dataset_name),
                        args=config,
                        callback=lambda x: pbar.update(),
                        error_callback=print_error,
                    )

                pool.close()
                pool.join()


def zip_csv(csv):
    with zipfile.ZipFile(f"{csv}.zip", "w", compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(csv)
    os.remove(csv)


def get_config_product(*args, repeat=1):
    iterables = [arg for arg in args if isinstance(arg, (tuple, list))]
    non_iterables = [arg for arg in args if not isinstance(arg, (tuple, list))]

    product_iterables = itertools.product(*iterables, repeat=repeat)

    result = []
    for p in product_iterables:
        result.append(tuple(non_iterables) + p)

    return result
