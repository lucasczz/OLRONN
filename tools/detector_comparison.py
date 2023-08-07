import copy
from itertools import product
from pathlib import Path
from src.models.drift_detectors import Manual, OneTailedADWIN, OneTailedPKSWIN
from src.models.schedulers import (
    ChainedScheduler,
    LRLimiter,
)
from torch.optim.lr_scheduler import ExponentialLR

from river.drift import ADWIN, KSWIN

from tools.base import (
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
run_name = "manual_and_wreset"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")


# Experiment config
lrs = [2**-i for i in range(-1, 9)]

gammas = [1 - 2**-i for i in range(12, 15)]
drift_confidence = 2e-4
detectors = [
    ('None', )
    ("ADWIN", ADWIN(drift_confidence)),
    ("KSWIN", KSWIN(drift_confidence)),
    ("OneTailedADWIN", OneTailedADWIN(1e-4)),
    ("Ground Truth", Manual(BATCH_SIZE)),
]
detectors += [(f"P-KSWIN_{2**-i}", OneTailedPKSWIN(bias=2**-i)) for i in range(4)]
drift_points = {
    "Insects abrupt": [14352, 19500, 33240, 38682, 39510],
    "RBF abrupt": [10000],
}


configs = list(product(lrs, detectors, [False], gammas, SEEDS))
configs += list(
    product(
        lrs, [("ADWIN Weight Reset", ADWIN(drift_confidence))], [True], gammas, SEEDS
    )
)


def run(
    lr,
    drift_detector,
    reset_weights,
    gamma,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    detector_name, detector = drift_detector
    if detector_name == "Ground Truth":
        if dataset_name in drift_points:
            detector.change_points = drift_points[dataset_name]
        else:
            return
    detector = copy.deepcopy(detector)
    scheduler_fn = lambda optim: ChainedScheduler(
        [
            ExponentialLR(optim, gamma=gamma),
            LRLimiter(optim, min_lr=0.1),
        ]
    )
    run_prequential(
        data=data,
        batch_size=BATCH_SIZE,
        scheduler_fn=scheduler_fn,
        drift_detector=detector,
        reset_weights=reset_weights,
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
        gamma=gamma,
        resetter=detector_name,
    )


if __name__ == "__main__":
    test_configs(run, DATASET_NAMES, configs=configs, debug=False)
    zip_csv(save_path)  
