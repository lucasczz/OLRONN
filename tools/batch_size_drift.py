from itertools import product
import torch
from pathlib import Path

from tools.base import (
    run_prequential,
    test_configs,
    REPORTS_PATH,
    DEVICE,
    SEEDS,
    LRS,
    DATASET_NAMES,
    zip_csv,
)

# Set up logging path
run_name = "v1"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

# Set up configs
lrs = LRS
batch_sizes = [2**i for i in range(5)]
n_hidden_layerss = [1]
datasets = [f"RBF incr._{i*0.001}" for i in range(2)] + DATASET_NAMES

configs = list(product(batch_sizes, n_hidden_layerss, lrs, SEEDS))


def run(batch_size, n_hidden_layers, lr, seed, data, dataset_name, verbose=False):
    run_prequential(
        data=data,
        batch_size=batch_size,
        optim_fn=torch.optim.SGD,
        base_lr=lr,
        log_path=save_path,
        net_params={
            "n_hidden_layers": n_hidden_layers,
        },
        seed=seed,
        n_hidden_layers=n_hidden_layers,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
    )


if __name__ == "__main__":
    test_configs(run, dataset_names=datasets, configs=configs, debug=False)
    zip_csv(save_path)
