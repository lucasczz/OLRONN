from pathlib import Path
from src.models.mechanic import get_mechanic_sgd, mechanize

from src.models.schedulers import ChainedScheduler, DriftResetLR, LRLimiter
from pathlib import Path
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from src.models.optimizers import COCOB, WNGrad
from dadaptation import DAdaptSGD, DAdaptLion, DAdaptAdam
from torch.optim import SGD, Adam, Adagrad
from dog import DoG
from src.models.schedulers import ChainedScheduler, DriftResetLR, LRLimiter
from tools.base import (
    get_config_product,
    run_prequential,
    test_configs,
    REPORTS_PATH,
    DEVICE,
    SEEDS,
    zip_csv,
)
from tools.pretune import count_parameters

# Set up logging path
run_name = "v3"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")
batch_sizes = [4]
n_hidden_layerss = [1, 3]
n_hidden_unitss = [32, 64, 128, 256]


# Set up configs
def get_exp_reset(optim):
    return ChainedScheduler(
        [
            ExponentialLR(optim, gamma=0.999756),
            LRLimiter(optim, min_lr=0.1),
            DriftResetLR(optim, detection_confidence=2e-4),
        ]
    )


def get_step_reset(optim):
    return ChainedScheduler(
        [
            StepLR(optim, gamma=0.875, step_size=2000),
            LRLimiter(optim, min_lr=0.1),
            DriftResetLR(optim, detection_confidence=2e-4),
        ]
    )


schedulers = [
    None,
    ("Exponential reset", get_exp_reset),
]

optimizers = [
    ("SGD", SGD, schedulers, [0.5]),
    ("Adam", Adam, None, [2**-7]),
    ("COCOB", COCOB, None, [100]),
    ("WNGrad", WNGrad, None, [10 ** (-0.25 * i + 1.25) for i in [4]]),
    ("DAdaptSGD", DAdaptSGD, None, [1]),
    ("DoG", DoG, None, [1]),
    ("AdaGrad", Adagrad, None, [0.25]),
    ("Mechanic", get_mechanic_sgd, None, [0.01])
]

configs = []
for optimizer in optimizers:
    configs += get_config_product(
        *optimizer, batch_sizes, n_hidden_layerss, n_hidden_unitss, SEEDS
    )


def run(
    optim_name,
    optim_fn,
    scheduler,
    lr,
    batch_size,
    n_hidden_layers,
    n_hidden_units,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    if scheduler is not None:
        scheduler_name, scheduler_fn = scheduler
    else:
        scheduler_name = "Fixed"
        scheduler_fn = None
    run_prequential(
        data=data,
        batch_size=batch_size,
        optim_fn=optim_fn,
        scheduler_fn=scheduler_fn,
        base_lr=lr,
        log_path=save_path,
        net_params={
            "n_hidden_layers": n_hidden_layers,
            "hidden_features": n_hidden_units,
        },
        seed=seed,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
        optimizer=optim_name,
        schedule=scheduler_name,
    )


if __name__ == "__main__":
    test_configs(
        run, dataset_names=["RBF abrupt"], configs=configs, debug=False, n_processes=1
    )
    zip_csv(save_path)
