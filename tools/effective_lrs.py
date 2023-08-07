import pandas as pd
from torch import nn
import torch
from pathlib import Path
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from src.models.drift_detectors import OneTailedADWIN
from src.models.mechanic import get_mechanic_sgd
from src.models.optimizers import COCOB, WNGrad
from dadaptation import DAdaptSGD
from torch.optim import SGD, Adagrad, Adam
from dog import DoG
from src.models.schedulers import (
    ChainedScheduler,
    LRLimiter,
    get_cyclic_scheduler,
)

from tools.base import (
    run_prequential,
    test_configs,
    REPORTS_PATH,
    DEVICE,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "v1"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

torch.set_default_dtype(torch.float64)


configs = [
    ("SGD", "Fixed"),
    ("SGD", "Exponential"),
    ("SGD", "Exponential reset"),
    ("SGD", "Step"),
    ("SGD", "Step reset"),
    ("SGD", "Cyclic"),
    ("SGD", "Cyclic reset"),
    ("Adam", "Fixed"),
    ("Adam", "Fixed reset"),
    ("COCOB", "Fixed"),
    ("WNGrad", "Fixed"),
    ("DAdaptSGD", "Fixed"),
    ("DoG", "Fixed"),
    ("AdaGrad", "Fixed"),
    ("Mechanic", "Fixed"),
]

optims = {
    "SGD": SGD,
    "AdaGrad": Adagrad,
    "Adam": Adam,
    "COCOB": COCOB,
    "WNGrad": WNGrad,
    "DAdaptSGD": DAdaptSGD,
    "DoG": DoG,
    "Mechanic": get_mechanic_sgd,
}
scheduler = {
    "Fixed": ExponentialLR,
    "Exponential": ExponentialLR,
    "Step": StepLR,
    "Cyclic": get_cyclic_scheduler,
}


batch_sizes = [4]
n_hidden_layerss = [1]
datasets = ["Insects abrupt"]
configs = [(*config, seed) for config in configs for seed in SEEDS]
best_params = pd.read_csv(save_path.parent.joinpath("best_params_insects_abrupt.csv"))


def run(
    optim_name,
    schedule_name,
    seed,
    data,
    dataset_name,
    batch_size=4,
    n_hidden_layers=1,
    verbose=False,
):
    params = best_params[
        (best_params["optimizer"] == optim_name)
        & (best_params["schedule"] == schedule_name)
    ].to_dict("records")[0]
    base_scheduler = schedule_name.strip(" reset")
    if base_scheduler == "Cyclic":
        scheduler_fn = lambda optim: ChainedScheduler(
            [
                scheduler[base_scheduler](
                    optim, max_lr=params["maxlr"], step_size_up=4000
                ),
                LRLimiter(optim, min_lr=0.1),
            ]
        )
    elif base_scheduler == "Step":
        scheduler_fn = lambda optim: ChainedScheduler(
            [
                scheduler[base_scheduler](optim, gamma=params["gamma"], step_size=2000),
                LRLimiter(optim, min_lr=0.1),
            ]
        )
    elif base_scheduler == "Fixed": 
        scheduler_fn = None
    else:
        scheduler_fn = lambda optim: ChainedScheduler(
            [
                scheduler[base_scheduler](optim, gamma=params["gamma"]),
                LRLimiter(optim, min_lr=0.1),
            ]
        )

    lr =  None if optim_name in ['DoG', "COCOB", "DAdaptSGD"] else params["base_lr"]
    drift_confidence = params["drift_confidence"]
    drift_detector = (
        OneTailedADWIN(delta=drift_confidence) if drift_confidence > 0 else None
    )
    optim_fn = optims[optim_name]

    run_prequential(
        data=data,
        batch_size=batch_size,
        optim_fn=optim_fn,
        scheduler_fn=scheduler_fn,
        drift_detector=drift_detector,
        base_lr=lr,
        log_path=save_path,
        log_lr_norms=True,
        log_variables=["lr"],
        net_params={"n_hidden_layers": n_hidden_layers, "activation": nn.LeakyReLU},
        seed=seed,
        n_hidden_layers=n_hidden_layers,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
        optimizer=optim_name,
        schedule=schedule_name,
    )


if __name__ == "__main__":
    test_configs(
        run,
        dataset_names=datasets,
        configs=configs,
        debug=False,
        dtype=torch.float64,
        n_processes=2,
    )
    zip_csv(save_path)
