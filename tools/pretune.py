from torch.optim.lr_scheduler import StepLR, ExponentialLR
from functools import partial
from itertools import product
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler
from tqdm import tqdm
from base import (
    REPORTS_PATH,
    SEEDS,
    bce_with_logits,
    accuracy,
    test_configs,
    zip_csv,
)
from src.models.networks import get_mlp
from src.models.schedulers import ChainedScheduler, LRLimiter
from src.utils import ExperimentLogger

# Set up logging path
run_name = "scheduled_exp"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

lrs = [2**-i for i in range(6)]
batch_sizes = [4]
n_hidden_layerss = [1, 3]
n_hidden_unitss = [64, 128]
datasets = [
    "RBF incr.",
    "RBF static",
    "Insects incr.",
    "Insects gradual",
    "Insects abrupt",
    "Electricity",
    "Covertype",
]

gammas = [1 - 2**-i for i in range(12, 15)] + [1]

n_samples = [500, 1000]
configs = list(
    product(
        lrs, batch_sizes, n_hidden_layerss, n_hidden_unitss, n_samples, gammas, SEEDS
    )
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tune_prequential(
    lr,
    batch_size,
    n_hidden_layers,
    n_hidden_units,
    n_samples,
    gamma,
    seed,
    data,
    dataset_name,
    steps=10000,
    verbose=False,
):
    torch.manual_seed(seed)
    x_sample, y_sample = data[0]
    net = get_mlp(
        x_sample.shape[-1],
        y_sample.shape[-1],
        n_hidden_layers=n_hidden_layers,
        hidden_features=n_hidden_units,
    )
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    if gamma < 1:
        scheduler = ChainedScheduler(
            [
                ExponentialLR(optim, gamma=gamma),
                LRLimiter(optim, min_lr=0.1),
            ]
        )
    n_params = count_parameters(net)
    if n_samples > len(data):
        return
    data = Subset(data, range(n_samples))
    sampler = RandomSampler(data, num_samples=steps, replacement=True)
    dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    logger = ExperimentLogger(
        optim,
        metric_fns=[accuracy],
        save_path=save_path,
        logged_variables=["lr"],
        parameters={
            "base_lr": lr,
            "dataset": dataset_name,
            "batch_size": batch_size,
            "seed": seed,
            "n_hidden_layers": n_hidden_layers,
            "n_hidden_units": n_hidden_units,
            "n_params": n_params,
            "n_samples": n_samples,
            "gamma": gamma,
            "schedule": "Exp",
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
        logger.update(y=y, y_pred=y_pred, loss=loss)


if __name__ == "__main__":
    test_configs(
        test_func=tune_prequential,
        dataset_names=datasets,
        configs=configs,
        debug=False,
    )
    zip_csv(save_path)
