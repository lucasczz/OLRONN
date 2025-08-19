from pathlib import Path
from tqdm import tqdm
import torch
from contam_base import run, pretrain_autoencoder, get_missing_configs, handle_log_queue
from base import get_config_grid
from src.models.anom_filters import AEFilter
from src.data.contamination import get_contaminated_stream, get_tuning_data
import traceback
import multiprocessing as mp


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
    len_anomaly,
    threshold_quantile=0.95,
    threshold_type="sigmoid",
    steepness=20,
    pretrain_samples=2000,
    validation_samples=500,
    verbose=True,
    device="cpu",
    seed=42,
):
    ae_hparams = {
        "lr": 0.25 if dataset == "Insects abrupt" else 1,
        "n_hidden_layers": 1,
        "n_hidden_units": 64 if dataset == "Insects abrupt" else 512,
        "dropout": 0,
        "epochs": 8,
    }

    clf_hparams = {
        "lr": 2**-4 if dataset == "Insects abrupt" else 2**-5,
        "n_hidden_layers": 1,
        "n_hidden_units": 1024,
        "n_classes": 9 if dataset == "Rotated MNIST" else 2,
    }

    x_pre, y_pre = get_tuning_data(dataset, tuning_samples=pretrain_samples)
    x_pre = torch.tensor(x_pre, dtype=torch.float, device=device)

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

    x_val = torch.tensor(
        x_stream[:validation_samples], dtype=torch.float, device=device
    )

    anom_filter = AEFilter(
        model=ae,
        threshold_quantile=threshold_quantile,
        threshold_type=threshold_type,
        steepness=steepness,
        device=device,
    )
    anom_filter.calibrate(x_val)

    x_test = torch.tensor(
        x_stream[validation_samples:], dtype=torch.float, device=device
    )
    y_test = torch.tensor(
        y_stream[validation_samples:], dtype=torch.long, device=device
    )

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
    run_name = "contam_filtering_linear.jsonl"
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)

    device_list = ["cuda:0", "cuda:1"]  # or just ["cuda:0"] if only one
    num_workers = len(device_list)

    configs = []

    # configs += get_config_grid(
    #     **{
    #         "dataset": ["Covertype"],
    #         "anomaly_type": ["feature_swap", "gaussian_noise"],
    #         "p_anomaly": [0.02, 0.04, 0.08],
    #         "len_anomaly": [2],
    #         "threshold_quantile": [0.9, 0.95, 0.975, 0.9875],
    #         "steepness": [15, 30, 60, np.inf],
    #         "seed": [0, 1, 2, 3, 4],
    #     }
    # )
    # configs += get_config_grid(
    #     **{
    #         "dataset": ["Insects abrupt", "Rotated MNIST"],
    #         "anomaly_type": ["ood_class"],
    #         "p_anomaly": [0.02, 0.04, 0.08],
    #         "len_anomaly": [2],
    #         "threshold_quantile": [0.9, 0.95, 0.975, 0.9875],
    #         "steepness": [15, 30, 60, np.inf],
    #         "seed": [0, 1, 2, 3, 4],
    #     }
    # )
    # configs += get_config_grid(
    #     **{
    #         "dataset": [
    #             "Covertype",
    #             "Insects abrupt",
    #             "Rotated MNIST",
    #         ],
    #         "anomaly_type": [
    #             "ood_class",
    #             # "feature_swap",
    #             # "gaussian_noise",
    #         ],
    #         "p_anomaly": [0.02, 0.04, 0.08],
    #         "len_anomaly": [2],
    #         "threshold_quantile": [0.9],
    #         "steepness": [0],
    #         "seed": [0, 1, 2, 3, 4],
    #     }
    # )

    configs += get_config_grid(
        **{
            "dataset": ["Covertype", "Insects abrupt", "Rotated MNIST"],
            "anomaly_type": ["ood_class"],
            "p_anomaly": [0.02, 0.04, 0.08],
            "len_anomaly": [2],
            "threshold_quantile": [
                .5,
                # 0.9, 0.95, 0.975, 0.9875, 1.0
                ],
            "threshold_type": ["linear"],
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
        # run_with_filter(**config)
        config["device"] = device_list[i % len(device_list)]
        config["verbose"] = False

    run_configs_parallel(configs, num_workers, logpath)
