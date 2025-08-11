from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import json
from tqdm import tqdm
import torch
from contam_base import (
    get_contaminated_stream,
    get_tuning_data,
    pretrain_autoencoder,
)
from base import get_config_grid


def run_ae(
    dataset,
    anomaly_type,
    p_anomaly,
    len_anomaly,
    n_hidden_layers,
    n_hidden_units,
    dropout,
    lr,
    epochs,
    pretrain_samples=2000,
    validation_samples=500,
    verbose=True,
    device="cpu",
    seed=42,
):
    x_stream, y_stream, is_anom = get_contaminated_stream(
        dataset=dataset,
        anomaly_type=anomaly_type,
        p_anomaly=p_anomaly,
        len_anomaly=len_anomaly,
        tuning_samples=pretrain_samples,
        seed=seed,
    )

    xt = torch.tensor(x_stream, dtype=torch.float)

    ae, anom_scores = pretrain_autoencoder(
        xt,
        n_hidden_layers,
        n_hidden_units,
        dropout,
        lr=lr,
        epochs=1,
        device=device,
        predict=True,
        seed=seed,
        verbose=True,
    )

    return anom_scores, is_anom


if __name__ == "__main__":
    run_name = "ae_tune2.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    device = "cuda:0"
    anom_scores, is_anom = run_ae(
        dataset="Rotated MNIST",
        anomaly_type="ood_class",
        p_anomaly=0.02,
        len_anomaly=2,
        lr=0.5,
        n_hidden_units=256,
        n_hidden_layers=1,
        epochs=1,
        dropout=0,
    )
    roc_auc = roc_auc_score(is_anom, anom_scores)
    print(roc_auc)
    
