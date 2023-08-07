from functools import partial
from src.models.mechanic import get_mechanic_sgd, mechanize
from itertools import product
from torch import optim
from pathlib import Path
from src.models.optimizers import COCOB, DDoG, WNGrad
from hypergrad import SGDHD
from dadaptation import DAdaptSGD, DAdaptAdam, DAdaptLion
from torch.optim import SGD, Adagrad
from dog import DoG, PolynomialDecayAverager

from tools.base import (
    get_config_product,
    run_prequential,
    test_configs,
    DATASET_NAMES,
    REPORTS_PATH,
    DEVICE,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "DDoG"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")


# Set up configs
optimizers = [
    ("Adam", optim.Adam, [2**-i for i in range(3, 13)]),
    ("SGD", SGD, [2**-i for i in range(-1, 9)]),
    ("SGDHD", SGDHD, [2**-i for i in range(3, 13)]),
    ("COCOB", COCOB, [100]),
    ("WNGrad", WNGrad, [10 ** (-0.25 * i + 1.25) for i in range(0, 19, 2)]),
    ("DAdaptSGD", DAdaptSGD, [1]),
    ("DoG", DoG, [1]),
    ("DAdaptAdam", DAdaptAdam, [1]),
    ("DAdaptLion", DAdaptLion, [1]),
    ("AdaGrad", Adagrad, [2**-i for i in range(-1, 9)]),
    ("Mechanic", get_mechanic_sgd, [0.01]),
    ("DDoG", DDoG, [1]),
]


batch_sizes = [4]
n_hidden_layerss = [1]
datasets = DATASET_NAMES


configs = []
for optimizer in optimizers:
    configs += get_config_product(*optimizer, batch_sizes, n_hidden_layerss, SEEDS)


def run(
    optim_name,
    optim_fn,
    lr,
    batch_size,
    n_hidden_layers,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    run_prequential(
        data=data,
        batch_size=batch_size,
        optim_fn=optim_fn,
        base_lr=lr,
        log_path=save_path,
        net_params={"n_hidden_layers": n_hidden_layers},
        seed=seed,
        n_hidden_layers=n_hidden_layers,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
        optimizer=optim_name,
        schedule="Fixed",
    )


if __name__ == "__main__":
    test_configs(
        run,
        dataset_names=datasets,
        configs=configs,
        debug=False,
    )
    zip_csv(save_path)
