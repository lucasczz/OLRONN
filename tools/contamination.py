from pathlib import Path
import json
from tqdm import tqdm
import torch
from contam_base import run, get_contaminated_stream
from base import get_config_grid


def test_contamination(
    dataset, anomaly_type, p_anomaly, len_anomaly, model_params, seed, device
):
    x_contam, y_contam, is_anom = get_contaminated_stream(
        dataset=dataset,
        anomaly_type=anomaly_type,
        p_anomaly=p_anomaly,
        len_anomaly=len_anomaly,
        seed=seed,
    )
    xt, yt = (
        torch.tensor(x_contam, dtype=torch.float),
        torch.tensor(y_contam, dtype=torch.long),
    )
    preds = run(xt, yt, hparams=model_params, seed=seed, device=device, verbose=False)
    return preds, y_contam, is_anom


if __name__ == "__main__":
    device = "cuda:0"
    run_name = "contamination2.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    model_params = {"lr": 2**-5, "n_hidden_units": 1024, "n_hidden_layers": 1, "n_classes": 2}

    contamination_configs = get_config_grid(
        **{
            "dataset": "Covertype",
            "anomaly_type": [
                "ood_sample",
                "ood_class",
                "label_flip",
            ],
            "p_anomaly": [0.0, 0.02, 0.04, 0.08],
            "seed": [0, 1, 2, 3, 4],
            "len_anomaly": [2, 4, 8, 16],
        }
    )
    for config in tqdm(contamination_configs, desc="Running configs"):
        preds, labels, is_anom = test_contamination(
            **config, model_params=model_params, device=device
        )
        result = config | {
            "preds": preds,
            "labels": labels.tolist(),
            "is_anom": is_anom.tolist(),
        }
        with open(logpath, "a") as f:
            f.write(json.dumps(result) + "\n")
