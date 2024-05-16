from functools import partial
from pathlib import Path
from src.models.drift_detectors import OneTailedKSWIN

from src.models.schedulers import (
    DriftResetLR,
    WeightResetLR,
    get_cyclic_lr,
)
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR


from tools.base import (
    DATASET_NAMES,
    DATASETS_SYNTH,
    DATASETS_REAL,
    LRS,
    REPORTS_PATH,
    SEEDS,
    get_config_grid,
    run_configs,
    zip_csv,
)

# Set up logging path
run_name = "v1"
log_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

schedules = [
    *[
        {
            "schedule": f"Exponential Reset",
            "scheduler_fn": [partial(ExponentialLR, gamma=1 - 2**-i), DriftResetLR],
            "gamma": 1 - 2**-i,
        }
        for i in range(12, 15)
    ],
    *[
        {
            "schedule": f"Exponential Reset (KSWIN)",
            "scheduler_fn": [
                partial(ExponentialLR, gamma=1 - 2**-i),
                partial(DriftResetLR, drift_detector_fn=OneTailedKSWIN),
            ],
            "gamma": 1 - 2**-i,
        }
        for i in range(12, 15)
    ],
    *[
        {
            "schedule": f"Exponential Weight Reset",
            "scheduler_fn": [partial(ExponentialLR, gamma=1 - 2**-i), WeightResetLR],
            "gamma": 1 - 2**-i,
        }
        for i in range(12, 15)
    ],
    {
        "schedule": "Fixed",
        "scheduler_fn": None,
    },
    *[
        {
            "schedule": "Exponential",
            "scheduler_fn": partial(ExponentialLR, gamma=1 - 2**-i),
            "gamma": 1 - 2**-i,
        }
        for i in range(12, 15)
    ],
    *[
        {
            "schedule": "Step",
            "scheduler_fn": partial(StepLR, gamma=g, step_size=2000),
            "gamma": g,
        }
        for g in [0.5, 0.75, 0.875]
    ],
    *[
        {
            "schedule": f"Cyclic",
            "scheduler_fn": partial(get_cyclic_lr, max_lr=g, step_size_up=4000),
            "max_lr": g,
        }
        for g in [0.125, 0.25, 0.5]
    ],
    *[
        {
            "schedule": f"Plateau",
            "scheduler_fn": partial(ReduceLROnPlateau, factor=1 - 2**-i, patience=400),
            "factor": 1 - 2**-i,
        }
        for i in range(5, 8)
    ],
]

configs = get_config_grid(
    schedules,
    base_lr=LRS,
    seed=SEEDS,
    optimizer="SGD",
    log_variables="lr",
    fixed_kwargs=dict(
        gamma=1,
        factor=1,
        max_lr=1,
    ),
)

if __name__ == "__main__":
    run_configs(
        dataset_names=DATASET_NAMES,
        configs=configs,
        debug=False,
        log_path=log_path,
    )
    zip_csv(log_path)
