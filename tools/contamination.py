from pathlib import Path
import traceback
import torch
from contam_base import run, get_missing_configs, run_configs_parallel
from base import get_config_grid
from src.data.contamination import get_contaminated_stream


def test_contamination(dataset, anomaly_type, p_anomaly, len_anomaly, seed, device):
    model_params = {
        "lr": 2**-4 if dataset == "Insects abrupt" else 2**-5,
        "n_hidden_units": 1024,
        "n_hidden_layers": 1,
        "n_classes": 9 if dataset == "Rotated MNIST" else 2,
    }
    x_contam, y_contam, is_anom = get_contaminated_stream(
        dataset=dataset,
        anomaly_type=anomaly_type,
        p_anomaly=p_anomaly,
        len_anomaly=len_anomaly,
        seed=seed,
    )
    xt, yt = (
        torch.tensor(x_contam, dtype=torch.float, device=device),
        torch.tensor(y_contam, dtype=torch.long, device=device),
    )
    preds = run(xt, yt, hparams=model_params, seed=seed, device=device, verbose=False)
    return preds, y_contam, is_anom


def run_config(config):
    try:
        preds, labels, is_anom = test_contamination(**config)
        return config | {
            "preds": preds,
            "labels": labels.tolist(),
            "is_anom": is_anom.tolist(),
        }
    except Exception as e:
        print("Error: ", config)
        print(traceback.format_exc())
        print(e)
        return None


if __name__ == "__main__":
    devices = ["cuda:0", "cuda:1"]
    num_workers = 4
    run_name = "contamination_fix.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    configs = get_config_grid(
        **{
            "dataset": ["Insects abrupt", "Covertype", "Rotated MNIST"],
            "anomaly_type": [
                "ood_sample",
                "ood_class",
                "label_flip",
                "feature_swap",
                "gaussian_noise",
            ],
            "p_anomaly": [0.02, 0.04, 0.08],
            "seed": [0, 1, 2, 3, 4],
            "len_anomaly": [2, 4, 8, 16],
        }
    )

    for i, config in enumerate(configs):
        config["device"] = devices[i % len(devices)]

    configs = get_missing_configs(
        configs,
        logpath,
        relevant_params=["dataset", "anomaly_type", "len_anomaly", "p_anomaly", "seed"],
    )

    run_configs_parallel(configs, run_config, num_workers, logpath)
