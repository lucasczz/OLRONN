from src.models.mechanic import get_mechanic_sgd
from pathlib import Path
from src.models.optimizers import CBP, COCOB, SGD_GC, SRSGD, DoWG, FirstOrderGlobalUPGD, RAdam, Ranger, WNGrad
from hypergrad import SGDHD
from dadaptation import DAdaptSGD
from torch.optim import SGD, Adagrad, Adam
from dog import DoG
from lomo_optim import AdaLomo
from came_pytorch import CAME
from lion_pytorch import Lion
from prodigyopt import Prodigy
from adabelief_pytorch import AdaBelief
from pbSGD import pbSGD

from tools.base import (
    DATASET_NAMES,
    DATASETS_SYNTH,
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
    {"optimizer": "DoWG", "optim_fn": DoWG, "base_lr": 1},
    {"optimizer": "Prodigy", "optim_fn": Prodigy, "base_lr": 1},
    {"optimizer": "Lion", "optim_fn": Lion, "base_lr": [2**-i for i in range(6, 16)]},
    {"optimizer": "CBP", "optim_fn": CBP, "base_lr": [2**-i for i in range(5, 15)]},
    {
        "optimizer": "UPGD",
        "optim_fn": FirstOrderGlobalUPGD,
        "base_lr": [2**-i for i in range(3, 13)],
    },
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
        dataset_names=DATASET_NAMES,
        configs=configs,
        debug=False,
        log_path=log_path,
        n_processes=1,
    )
    zip_csv(log_path)
