from src.models.mechanic import get_mechanic_sgd
from pathlib import Path
from src.models.optimizers import COCOB, WNGrad
from hypergrad import SGDHD
from dadaptation import DAdaptSGD
from torch.optim import SGD, Adagrad, Adam
from dog import DoG

from tools.base import (
    DATASET_NAMES,
    get_config_grid,
    run_configs,
    REPORTS_PATH,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "v1"
log_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

# Set up configs
optimizers = [
    {"optimizer": "SGD", "optim_fn": SGD, "base_lr": [2**-i for i in range(-1, 9)]},
    {"optimizer": "Adam", "optim_fn": Adam, "base_lr": [2**-i for i in range(3, 13)]},
    {
        "optimizer": "SGDHD",
        "optim_fn": SGDHD,
        "base_lr": [2**-i for i in range(3, 13)],
    },
    {"optimizer": "COCOB", "optim_fn": COCOB, "base_lr": 100},
    {"optimizer": "DAdaptSGD", "optim_fn": DAdaptSGD, "base_lr": 1},
    {"optimizer": "DoG", "optim_fn": DoG, "base_lr": 1},
    {"optimizer": "Mechanic", "optim_fn": get_mechanic_sgd, "base_lr": 0.01},
    {
        "optimizer": "AdaGrad",
        "optim_fn": Adagrad,
        "base_lr": [2**-i for i in range(-1, 9)],
    },
    {
        "optimizer": "WNGrad",
        "optim_fn": WNGrad,
        "base_lr": [10 ** (-i / 2 + 1.25) for i in range(10)],
    },
]


configs = get_config_grid(optimizers, seed=SEEDS, log_lr_norms=True)

if __name__ == "__main__":
    run_configs(
        dataset_names=DATASET_NAMES, configs=configs, debug=False, log_path=log_path
    )
    zip_csv(log_path)
