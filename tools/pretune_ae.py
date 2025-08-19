import json
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from base import get_config_grid
from contam_base import pretrain_autoencoder, run_configs_parallel, get_missing_configs
from tqdm import tqdm

from src.data.contamination import get_contaminated_stream, get_tuning_data
from src.models.networks import get_autoencoder


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
    lr_online=0.5,
    pretrain_samples=2000,
    validation_samples=500,
    finetuning_steps=2000,
    online_finetuning=False,
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
    optimizer = None

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

    # np.random.seed(seed)
    # idcs_resample = np.random.choice(np.arange(len(is_anom)), size=finetuning_steps)
    # x_stream = x_stream[idcs_resample]
    # is_anom = is_anom[idcs_resample]

    xt = torch.tensor(x_stream, dtype=torch.float, device=device)

    if not online_finetuning:
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
                    dropout=dropout,
                    n_hidden_units=n_hidden_units,
                    n_hidden_layers=n_hidden_layers,
                )
                ae = ae.to(device)
            if optimizer is None:
                optimizer = torch.optim.SGD(ae.parameters(), lr=lr * lr_online)

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
    run_name = "ae_tune_16.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    devices = ["cuda:0", "cuda:1"]
    num_workers = 6
    configs = []
    # configs += get_config_grid(
    #     **{
    #         "n_hidden_units": [64, 128, 256, 512],
    #         "n_hidden_layers": [1, 2],
    #         "dropout": 0.0,
    #         "epochs": [8],
    #         "lr": [2**-i for i in range(6)],
    #         "lr_online": [0.25, 0.5, 1.0, 2.0],
    #         "online_finetuning": True,
    #         "dataset": ["Insects abrupt", "Rotated MNIST", "Covertype"],
    #         "anomaly_type": "ood_class",
    #         "p_anomaly": 0.04,
    #         "len_anomaly": 2,
    #     }
    # )
    # configs += get_config_grid(
    #     **{
    #         "n_hidden_units": [64, 128, 256, 512],
    #         "n_hidden_layers": [1, 2],
    #         "dropout": 0.0,
    #         "epochs": 0,
    #         "lr": 1,
    #         "lr_online": [2**-i for i in range(6)],
    #         "online_finetuning": True,
    #         "dataset": ["Insects abrupt", "Rotated MNIST", "Covertype"],
    #         "anomaly_type": "ood_class",
    #         "p_anomaly": 0.04,
    #         "len_anomaly": 2,
    #     }
    # )
    configs += get_config_grid(
        **{
            "n_hidden_units": [64, 128, 256, 512],
            "n_hidden_layers": [1, 2],
            "dropout": 0.0,
            "epochs": 16,
            "lr": [2**-i for i in range(6)],
            "lr_online": 1.0,
            "online_finetuning": False,
            "dataset": ["Insects abrupt", "Rotated MNIST", "Covertype"],
            "anomaly_type": "ood_class",
            "p_anomaly": 0.04,
            "len_anomaly": 2,
        }
    )

    # configs = get_missing_configs(
    #     configs,
    #     logpath,
    #     relevant_params=[
    #         "n_hidden_units",
    #         "n_hidden_layers",
    #         "lr",
    #         "lr_online",
    #         "dataset",
    #     ],
    # )

    # Append constant hparams and device info
    for i, config in enumerate(configs):
        # run_with_filter(**config)
        config["device"] = devices[i % len(devices)]

    run_configs_parallel(configs, run_config, num_workers, logpath)
