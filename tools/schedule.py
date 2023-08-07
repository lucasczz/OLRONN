from functools import partial
from river.drift import ADWIN
from torch import optim
from pathlib import Path
from dog import DoG
from src.models.drift_detectors import OneTailedADWIN

from src.models.schedulers import (
    DriftResetLR,
    ChainedScheduler,
    LRLimiter,
    get_cyclic_scheduler,
)
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR


from tools.base import (
    get_config_product,
    run_prequential,
    test_configs,
    DATASET_NAMES,
    REPORTS_PATH,
    DEVICE,
    BATCH_SIZE,
    N_HIDDEN_LAYERS,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "new_reset"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")


# Experiment config
optimizers = [
    ("SGD", optim.SGD, [2**-i for i in range(-1, 9)]),
    ("Adam", optim.Adam, [2**-i for i in range(3, 13)]),
    ("DoG", DoG, [1]),
]

plateau_schedulers = [
    (f"Plateau_factor={f}", partial(ReduceLROnPlateau, factor=f, patience=400))
    for f in [0.9375, 0.96875, 1 - 2**-6]
]
exp_schedulers = [
    (f"Exponential_gamma={g}", partial(ExponentialLR, gamma=g))
    for g in [1] + [1 - 2**-i for i in range(12, 15)]
]
step_schedulers = [
    (f"Step_gamma={g}", partial(StepLR, gamma=g, step_size=2000))
    for g in [0.5, 0.75, 0.875]
]
cyclic_schedulers = [
    (f"Cyclic_maxlr={l}", partial(get_cyclic_scheduler, max_lr=l, step_size_up=4000))
    for l in [0.25, 0.5]
]

schedulers = exp_schedulers + step_schedulers + cyclic_schedulers
drift_confidences = [1e-4]


configs = []
for optimizer in optimizers:
    configs += get_config_product(*optimizer, schedulers, drift_confidences, SEEDS)


def run(
    optimizer_name,
    optimizer_fn,
    lr,
    base_scheduler,
    drift_confidence,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    scheduler_name, scheduler = base_scheduler
    drift_detector = (
        OneTailedADWIN(delta=drift_confidence) if drift_confidence > 0 else None
    )
    scheduler_fn = lambda optim: ChainedScheduler(
        [scheduler(optim), LRLimiter(optim, min_lr=0.1)]
    )
    run_prequential(
        data=data,
        batch_size=BATCH_SIZE,
        optim_fn=optimizer_fn,
        optimizer=optimizer_name,
        scheduler_fn=scheduler_fn,
        drift_detector=drift_detector,
        base_lr=lr,
        log_path=save_path,
        log_variables=["lr"],
        net_params={"n_hidden_layers": N_HIDDEN_LAYERS},
        seed=seed,
        n_hidden_layers=N_HIDDEN_LAYERS,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
        drift_confidence=drift_confidence,
        schedule=scheduler_name,
    )


if __name__ == "__main__":
    test_configs(run, DATASET_NAMES, configs=configs, debug=False, n_processes=2)
    zip_csv(save_path)
