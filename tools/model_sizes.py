import copy
from itertools import product
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from pathlib import Path
from src.models.drift_detectors import OneTailedADWIN
from tools.pretune import (
    datasets,
    count_parameters,
)
from src.models.networks import get_mlp
from src.models.schedulers import ChainedScheduler, LRLimiter
from tools.pretune import batch_sizes, n_hidden_layerss, n_hidden_unitss, gammas, lrs

from tools.base import (
    run_prequential,
    test_configs,
    REPORTS_PATH,
    DEVICE,
    SEEDS,
    zip_csv,
)

# Set up logging path
run_name = "v3"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")
drift_detectors = [None, OneTailedADWIN(1e-4)]

configs = list(
    product(
        batch_sizes,
        n_hidden_layerss,
        n_hidden_unitss,
        lrs,
        drift_detectors,
        gammas,
        SEEDS,
    )
)


def run(
    batch_size,
    n_hidden_layers,
    n_hidden_units,
    lr,
    drift_detector,
    gamma,
    seed,
    data,
    dataset_name,
    verbose=False,
):
    sample_x, sample_y = data[0]
    in_features, out_features = sample_x.shape[-1], sample_y.shape[-1]
    net = get_mlp(in_features, out_features, n_hidden_units, n_hidden_layers)
    n_params = count_parameters(net)

    scheduler_fn = lambda optim: ChainedScheduler(
        [
            ExponentialLR(optim, gamma=gamma),
            LRLimiter(optim, min_lr=0.1),
        ]
    )
    
    if drift_detector is not None:
        drift_detector = copy.deepcopy(drift_detector)
        drift_confidence = 1e-4
    else:
        drift_confidence = 0
    run_prequential(
        data=data,
        batch_size=batch_size,
        optim_fn=torch.optim.SGD,
        base_lr=lr,
        scheduler_fn=scheduler_fn,
        drift_detector=drift_detector,
        log_path=save_path,
        net_params={
            "n_hidden_layers": n_hidden_layers,
            "hidden_features": n_hidden_units,
        },
        gamma=gamma,
        seed=seed,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        n_params=n_params,
        verbose=verbose,
        device=DEVICE,
        dataset=dataset_name,
        drift_confidence=drift_confidence
    )


if __name__ == "__main__":
    test_configs(
        run, dataset_names=datasets, configs=configs, debug=False, subsample=1000
    )
    zip_csv(save_path)
