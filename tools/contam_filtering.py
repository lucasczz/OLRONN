from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import torch
from contam_base import run, pretrain_autoencoder
from base import get_config_grid
from src.models.anom_filters import AEFilter
from src.data.contamination import get_contaminated_stream, get_tuning_data


def run_with_filter(
    dataset,
    anomaly_type,
    p_anomaly,
    len_anomaly,
    ae_hparams,
    clf_hparams,
    threshold_quantile=0.95,
    steepness=20,
    pretrain_samples=2000,
    validation_samples=500,
    verbose=True,
    device="cpu",
    seed=42,
):
    x_pre, y_pre = get_tuning_data(dataset, tuning_samples=pretrain_samples)
    x_pre = torch.tensor(x_pre, dtype=torch.float)

    ae = pretrain_autoencoder(
        x_pre,
        ae_hparams["n_hidden_layers"],
        ae_hparams["n_hidden_units"],
        ae_hparams["dropout"],
        lr=ae_hparams["lr"],
        epochs=ae_hparams["epochs"],
        device=device,
        seed=seed,
        verbose=verbose,
    )

    x_stream, y_stream, is_anom = get_contaminated_stream(
        dataset=dataset,
        anomaly_type=anomaly_type,
        p_anomaly=p_anomaly,
        len_anomaly=len_anomaly,
        tuning_samples=pretrain_samples,
        seed=seed,
    )

    x_val = torch.tensor(x_stream[:validation_samples], dtype=torch.float)

    anom_filter = AEFilter(
        model=ae,
        threshold_quantile=threshold_quantile,
        steepness=steepness,
        device=device,
    )
    anom_filter.calibrate(x_val)

    x_test = torch.tensor(x_stream[validation_samples:], dtype=torch.float)
    y_test = torch.tensor(y_stream[validation_samples:], dtype=torch.long)

    preds, loss_weights = run(
        x_test, y_test, clf_hparams, anom_filter, device, seed, verbose
    )
    return (
        preds,
        y_stream[validation_samples:],
        loss_weights,
        is_anom[validation_samples:],
    )


if __name__ == "__main__":
    run_name = "contam_filtering_grid.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    device = "cuda:0"

    ae_hparams = {
        "lr": 1,
        "n_hidden_layers": 1,
        "n_hidden_units": 64,
        "dropout": 0,
        "epochs": 8,
    }

    clf_hparams = {
        "lr": 2**-5,
        "n_hidden_layers": 1,
        "n_hidden_units": 1024,
        "n_classes": 2,
    }

    configs = get_config_grid(
        **{
            "dataset": "Covertype",
            "anomaly_type": "ood_class",
            "p_anomaly": [0.02, 0.04, 0.08],
            "len_anomaly": [2],
            "threshold_quantile": [0.9, 0.95, 0.975, 0.9875],
            "steepness": [0, 15, 30, 60, np.inf],
            "seed": [0, 1, 2, 3, 4],
        }
    )

    for config in tqdm(configs, desc="Running configs"):
        preds, labels, loss_weights, is_anom = run_with_filter(
            **config,
            clf_hparams=clf_hparams,
            ae_hparams=ae_hparams,
            device=device,
            verbose=False,
        )
        result = config | {
            "preds": preds,
            "labels": labels.tolist(),
            "loss_weights": loss_weights,
            "is_anom": is_anom.tolist(),
        }
        with open(logpath, "a") as f:
            f.write(json.dumps(result) + "\n")
