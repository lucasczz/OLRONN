from pathlib import Path
import traceback
import numpy as np
import torch.nn.functional as F
import torch
from src.models.networks import get_autoencoder
from contam_base import run_configs_parallel, get_missing_configs
from src.models.anom_filters import pretrain_autoencoder

from src.data.contamination import get_contaminated_stream, get_tuning_data
from base import get_config_grid

CONFIGS = {
    "Covertype": {
        "lr": 1,
        "lr_online": 0.25,
        "lr_finetuning": 0.5,
        "n_hidden_layers": 1,
        "n_hidden_units": 64,
    },
    "Insects abrupt": {
        "lr": 0.0625,
        "lr_online": 0.125,
        "lr_finetuning": 0.125,
        "n_hidden_layers": 1,
        "n_hidden_units": 512,
    },
    "Rotated MNIST": {
        "lr": 1.0,
        "lr_online": 1,
        "lr_finetuning": 0.125,
        "n_hidden_layers": 2,
        "n_hidden_units": 64,
    },
}


def run_ae(
    dataset,
    anomaly_type,
    p_anomaly,
    len_anomaly,
    epochs=8,
    mode="pre-trained",
    lr=0.5,
    lr_online=0.125,
    lr_finetuning=0.125,
    n_hidden_layers=1,
    n_hidden_units=64,
    pretrain_samples=2000,
    validation_samples=500,
    verbose=True,
    device="cpu",
    seed=42,
):
    x_pre, y_pre = get_tuning_data(dataset, tuning_samples=pretrain_samples)
    x_pre = torch.tensor(x_pre, dtype=torch.float)

    _lr_online = lr_online if mode == "online" else lr_finetuning
    torch.manual_seed(seed)

    if mode in ["pre-trained", "pre-trained+online"]:
        ae = pretrain_autoencoder(
            x_pre,
            n_hidden_layers,
            n_hidden_units,
            lr=lr,
            epochs=epochs,
            device=device,
            verbose=verbose,
        )
    else:
        ae = None

    optimizer = None
    x_stream, y_stream, is_anom = get_contaminated_stream(
        dataset=dataset,
        anomaly_type=anomaly_type,
        p_anomaly=p_anomaly,
        len_anomaly=len_anomaly,
        tuning_samples=pretrain_samples,
        seed=seed,
    )

    x_stream = x_stream[validation_samples:]
    is_anom = is_anom[validation_samples:]

    xt = torch.tensor(x_stream, dtype=torch.float, device=device)

    if mode == "pre-trained":
        ae.eval()
        with torch.inference_mode():
            xt = xt.to(device)
            x_rec = ae(xt)
            anom_scores = F.l1_loss(xt, x_rec, reduction="none").mean(dim=-1)

        anom_scores = anom_scores.numpy(force=True)
    else:
        anom_scores = []
        for xti in xt:
            if ae is None:
                ae = get_autoencoder(
                    in_features=xti.shape[-1],
                    n_hidden_units=n_hidden_units,
                    n_hidden_layers=n_hidden_layers,
                )
                ae = ae.to(device)
            if optimizer is None:
                optimizer = torch.optim.SGD(ae.parameters(), lr=_lr_online)

            ae.eval()
            with torch.inference_mode():
                x_rec = ae(xti)
                anom_score = F.l1_loss(xti, x_rec).detach().cpu().item()
                anom_scores.append(anom_score)

            ae.train()
            x_rec = ae(xti)
            loss = F.l1_loss(xti, x_rec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        anom_scores = np.array(anom_scores)

    return anom_scores, is_anom


def run_config(config):
    try:
        anom_scores, is_anom = run_ae(**config, verbose=False)
        return config | {
            "anom_scores": anom_scores.tolist(),
            "is_anom": is_anom.tolist(),
        }
    except Exception as e:
        print("Error: ", config)
        print(traceback.format_exc())
        print(e)
        return None


if __name__ == "__main__":
    run_name = "ae_test_v4.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    devices = ["cuda:0", "cuda:1"]
    num_workers = 4

    configs = get_config_grid(
        **{
            "dataset": ["Covertype", "Insects abrupt", "Rotated MNIST"],
            "mode": [
                "pre-trained",
                "online",
                "pre-trained+online",
            ],
            "anomaly_type": [
                # "ood_class",
                "feature_swap",
            ],
            "p_anomaly": [
                0.02,
                0.04,
                0.08,
            ],
            "len_anomaly": [2],
            "seed": [0, 1, 2, 3, 4],
        }
    )

    configs += get_config_grid(
        **{
            "dataset": ["Covertype", "Insects abrupt", "Rotated MNIST"],
            "mode": ["online"],
            "anomaly_type": [
                "ood_class",
                "feature_swap",
            ],
            "p_anomaly": [0.04],
            "len_anomaly": [2, 4, 8, 16],
            "seed": [0, 1, 2, 3, 4],
        }
    )

    # Append constant hparams and device info
    updated_configs = []
    for i, config in enumerate(configs):
        config["device"] = devices[i % len(devices)]
        new_config = config | (CONFIGS[config["dataset"]])
        updated_configs.append(new_config)

    updated_configs = get_missing_configs(
        updated_configs,
        logpath,
        relevant_params=["dataset", "mode", "len_anomaly", "p_anomaly", "seed"],
    )

    run_configs_parallel(updated_configs, run_config, num_workers, logpath)
