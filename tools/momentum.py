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
    LRS,
    zip_csv,
)

# Set up logging path
run_name = "nesterov"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

# Set up configs
lrs = [2**-i for i in range(6)]
batch_sizes = [4, 16]
net_depths = [1, 3]
datasets = ["Insects incr.", "Electricity", "Insects abrupt", "RBF abrupt"]
momentums = [0, 0.25, 0.5, 0.75]
nesterovs = [True]


configs = list(
    product(
        [optim.SGD],
        momentums,
        nesterovs,
        batch_sizes,
        net_depths,
        lrs,
        SEEDS,
    )
)


def run(
    optimizer,
    momentum,
    nesterov,
    batch_size,
    net_depth,
    lr,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    optimizer_fn = partial(optimizer, momentum=momentum, nesterov=nesterov)
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
        weight_decay=0,
        momentum=momentum,
        nesterov=nesterov,
    )


if __name__ == "__main__":
    test_configs(run, dataset_names=datasets, configs=configs)
    zip_csv(save_path)
