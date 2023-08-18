from pathlib import Path
from torch.optim import SGD
from dog import DoG
from tools.pretune import datasets
from tools.pretune import gamma, n_hidden_layers, n_hidden_units, lr

from tools.base import (
    get_config_grid,
    run_configs,
    REPORTS_PATH,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "v1"
log_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

optimizer = [
    {"optimizer": "DoG", "optim_fn": DoG, "base_lr": 1},
    {"optimizer": "SGD", "optim_fn": SGD, "base_lr": lr},
]
configs = get_config_grid(
    optimizer,
    n_hidden_layers=n_hidden_layers,
    n_hidden_units=n_hidden_units,
    gamma=gamma,
    subsample=1000,
    seed=SEEDS,
)


if __name__ == "__main__":
    run_configs(dataset_names=datasets, configs=configs, debug=True, log_path=log_path)
    zip_csv(log_path)
