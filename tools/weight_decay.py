from functools import partial
from itertools import product
from torch import optim
from pathlib import Path

from tools.base import (
    run_prequential,
    test_configs,
    DATASET_NAMES,
    REPORTS_PATH,
    DEVICE,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "v3"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

# Set up configs
lrs = [2**-i for i in range(6)]
batch_sizes = [4, 16]
net_depths = [1, 3]
datasets = ["Insects incr.", "Electricity", "Insects abrupt", "RBF abrupt"]
weight_decays = [0, 1e-5, 1e-4, 1e-3]


configs = list(
    product(
        [optim.SGD],
        weight_decays,
        batch_sizes,
        net_depths,
        lrs,
        SEEDS,
    )
)


def run(
    optimizer,
    weight_decay,
    batch_size,
    net_depth,
    lr,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    optimizer_fn = partial(optimizer, weight_decay=weight_decay)
    run_prequential(
        data=data,
        batch_size=batch_size,
        optim_fn=optimizer_fn,
        base_lr=lr,
        log_path=save_path,
        net_params={"n_hidden_layers": net_depth},
        seed=seed,
        n_hidden_layers=net_depth,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
        weight_decay=weight_decay,
        momentum=0,
    )


if __name__ == "__main__":
    test_configs(run, dataset_names=datasets, configs=configs)
    zip_csv(save_path)
