from pathlib import Path
from tqdm import tqdm
import torch
from contam_base import run, get_missing_configs, handle_log_queue
from base import get_config_grid
from src.models.anom_filters import AEFilter
from src.data.contamination import get_contaminated_stream, get_tuning_data
import traceback
import multiprocessing as mp

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


def run_configs_parallel(configs, num_workers, file_path):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    q = manager.Queue()

    pool = mp.Pool(processes=num_workers)

    queue_process = mp.Process(target=handle_log_queue, args=(q, file_path))
    queue_process.start()

    for result in tqdm(pool.imap(run_with_filter_kwargs, configs), total=len(configs)):
        if result is not None:
            q.put(result)

    q.put("kill")
    pool.close()
    pool.join()
    queue_process.join()


def run_with_filter_kwargs(kwargs):
    try:
        preds, labels, loss_weights, is_anom = run_with_filter(**kwargs)
        return kwargs | {
            "preds": preds,
            "labels": labels.tolist(),
            # "loss_weights": loss_weights,
            "is_anom": is_anom.tolist(),
        }
    except Exception as e:
        print("Error: ", kwargs)
        print(traceback.format_exc())
        print(e)
        return None


def run_with_filter(
    dataset,
    anomaly_type,
    p_anomaly,
    len_anomaly=2,
    threshold_quantile=None,
    threshold_type="sigmoid",
    threshold_win_size=1000,
    epochs=8,
    mode="pre-trained",
    lr=0.5,
    lr_online=0.125,
    lr_finetuning=0.125,
    n_hidden_layers=1,
    n_hidden_units=64,
    steepness=20,
    pretrain_samples=2000,
    validation_samples=500,
    verbose=True,
    device="cpu",
    seed=42,
):
    if mode == "pre-trained":
        _lr_online = 0
    elif mode == "pre-trained+online":
        _lr_online = lr_finetuning
    elif mode == "online":
        _lr_online = lr_online

    anom_filter = AEFilter(
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        lr_pretraining=lr,
        lr_online=_lr_online,
        epochs=epochs,
        threshold_quantile=1 - p_anomaly
        if threshold_quantile is None
        else threshold_quantile,
        threshold_type=threshold_type,
        window_size=threshold_win_size,
        steepness=steepness,
        device=device,
    )

    clf_hparams = {
        "lr": 2**-4 if dataset == "Insects abrupt" else 2**-5,
        "n_hidden_layers": 1,
        "n_hidden_units": 1024,
        "n_classes": 9 if dataset == "Rotated MNIST" else 2,
    }

    x_pre, y_pre = get_tuning_data(dataset, tuning_samples=pretrain_samples)
    x_pre = torch.tensor(x_pre, dtype=torch.float, device=device)

    x_stream, y_stream, is_anom = get_contaminated_stream(
        dataset=dataset,
        anomaly_type=anomaly_type,
        p_anomaly=p_anomaly,
        len_anomaly=len_anomaly,
        tuning_samples=pretrain_samples,
        seed=seed,
    )

    x_val = torch.tensor(
        x_stream[:validation_samples], dtype=torch.float, device=device
    )

    x_test = torch.tensor(
        x_stream[validation_samples:], dtype=torch.float, device=device
    )
    y_test = torch.tensor(
        y_stream[validation_samples:], dtype=torch.long, device=device
    )

    if mode in ["pre-trained", "pre-trained+online"]:
        anom_filter.pre_train(x_pre)
        anom_filter.calibrate(x_val)

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
    run_name = "contam_filtering_v2.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    device_list = ["cuda:0", "cuda:1"]  # or just ["cuda:0"] if only one
    num_workers = 4

    configs = []

    configs += get_config_grid(
        **{
            "dataset": [
                "Covertype",
                "Insects abrupt",
                "Rotated MNIST",
            ],
            "anomaly_type": [
                "feature_swap",
                "ood_class",
            ],
            "p_anomaly": [
                0.02,
                0.04,
                0.08,
            ],
            "len_anomaly": [2],
            "mode": [
                "pre-trained",
                # "online",
                # "pre-trained+online",
            ],
            "threshold_type": ["linear", "hard", "sigmoid"],
            "seed": [0, 1, 2, 3, 4],
        }
    )

    configs += get_config_grid(
        **{
            "dataset": [
                "Covertype",
                "Insects abrupt",
                "Rotated MNIST",
            ],
            "anomaly_type": [
                "feature_swap",
                "ood_class",
            ],
            "p_anomaly": [
                0.02,
                0.04,
                0.08,
            ],
            "len_anomaly": [2],
            "mode": [
                # "pre-trained",
                "online",
                "pre-trained+online",
            ],
            "threshold_type": [
                "linear",
                # "hard",
                # "sigmoid",
            ],
            "seed": [0, 1, 2, 3, 4],
        }
    )

    # configs = get_missing_configs(
    #     configs,
    #     logpath,
    #     relevant_params=[
    #         "dataset",
    #         "anomaly_type",
    #         "p_anomaly",
    #         "threshold_quantile",
    #         "steepness",
    #         "seed",
    #     ],
    # )

    # Append constant hparams and device info
    for i, config in enumerate(configs):
        config.update(CONFIGS[config["dataset"]])
        config["device"] = device_list[i % len(device_list)]
        config["verbose"] = False
        # run_with_filter(**config)

    run_configs_parallel(configs, num_workers, logpath)
