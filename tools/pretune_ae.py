from pathlib import Path
import torch.nn.functional as F
import json
from tqdm import tqdm
import torch
from contam_base import run, pretrain_autoencoder

from src.data.contamination import get_contaminated_stream, get_tuning_data
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
    x_pre, y_pre = get_tuning_data(dataset, tuning_samples=pretrain_samples)
    x_pre = torch.tensor(x_pre, dtype=torch.float)

    ae = pretrain_autoencoder(
        x_pre,
        n_hidden_layers,
        n_hidden_units,
        dropout,
        lr=lr,
        epochs=epochs,
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
    x_stream = x_stream[:validation_samples]
    is_anom = is_anom[:validation_samples]

    xt = torch.tensor(x_stream, dtype=torch.float)

    ae.eval()
    with torch.inference_mode():
        xt = xt.to(device)
        x_rec = ae(xt)
        anom_scores = F.l1_loss(xt, x_rec, reduction="none").mean(dim=-1)

    anom_scores = anom_scores.numpy(force=True)

    return anom_scores, is_anom


if __name__ == "__main__":
    run_name = "ae_tune.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    device = "cuda:0"

    ae_configs = get_config_grid(
        **{
            "n_hidden_units": [64, 128, 256, 512],
            "n_hidden_layers": [1, 2],
            "dropout": [0.0, 0.1],
            "epochs": [8],
            "lr": [2**-i for i in range(6)],
            "dataset": ["Insects abrupt"],
            "anomaly_type": "ood_class",
            "p_anomaly": 0.08,
            "len_anomaly": 2,
        }
    )

    for config in tqdm(ae_configs, desc="Running configs"):
        anom_scores, is_anom = run_ae(**config, device=device, verbose=False)
        result = config | {
            "anom_scores": anom_scores.tolist(),
            "is_anom": is_anom.tolist(),
        }
        with open(logpath, "a") as f:
            f.write(json.dumps(result) + "\n")
