from pathlib import Path
import traceback
import numpy as np
import torch.nn.functional as F
import torch
from src.models.networks import get_autoencoder
from contam_base import pretrain_autoencoder, run_configs_parallel

from src.data.contamination import get_contaminated_stream, get_tuning_data
from base import get_config_grid


def run_ae(
    dataset,
    anomaly_type,
    p_anomaly,
    len_anomaly,
    pre_training=True,
    online_finetuning=False,
    lr_online=1e-3,
    pretrain_samples=2000,
    validation_samples=500,
    verbose=True,
    device="cpu",
    seed=42,
):
    x_pre, y_pre = get_tuning_data(dataset, tuning_samples=pretrain_samples)
    x_pre = torch.tensor(x_pre, dtype=torch.float)

    ae_hparams = {
        "lr": 0.25 if dataset == "Insects abrupt" else 1,
        "n_hidden_layers": 1,
        "n_hidden_units": 64 if dataset == "Insects abrupt" else 512,
        "dropout": 0,
        "epochs": 8,
    }

    if pre_training:
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

    ae.eval()

    if not online_finetuning:
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
                    dropout=ae_hparams["dropout"],
                    n_hidden_units=ae_hparams["n_hidden_units"],
                    n_hidden_layers=ae_hparams["n_hidden_layers"],
                )
                ae = ae.to(device)
            if optimizer is None:
                optimizer = torch.optim.SGD(ae.parameters(), lr=lr_online)

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
    run_name = "ae_test.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    devices = ["cuda:0", "cuda:1"]
    num_workers = 2

    configs = get_config_grid(
        **{
            "dataset": ["Covertype", "Insects abrupt", "Rotated MNIST"],
            "lr_online": [2**-i for i in range(6)],
            "online_finetuning": True,
            "anomaly_type": "ood_class",
            "p_anomaly": [0.02, 0.04, 0.08],
            "len_anomaly": 2,
            "seed": [0, 1, 2, 3, 4],
        }
    )
    configs += get_config_grid(
        **{
            "dataset": ["Covertype", "Insects abrupt", "Rotated MNIST"],
            "lr_online": 0.5,
            "online_finetuning": False,
            "anomaly_type": "ood_class",
            "p_anomaly": 0.04,
            "len_anomaly": 2,
            "seed": [0, 1, 2, 3, 4],
        }
    )

    # Append constant hparams and device info
    for i, config in enumerate(configs):
        # run_with_filter(**config)
        config["device"] = devices[i % len(devices)]

    run_configs_parallel(configs, run_config, num_workers, logpath)
