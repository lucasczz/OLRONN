import torch
import json
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

import multiprocessing as mp
from src.models.networks import get_mlp, get_autoencoder


def run_configs_parallel(configs, target_fn, num_workers, file_path):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    q = manager.Queue()

    pool = mp.Pool(processes=num_workers)

    queue_process = mp.Process(target=handle_log_queue, args=(q, file_path))
    queue_process.start()

    for result in tqdm(pool.imap(target_fn, configs), total=len(configs)):
        if result is not None:
            q.put(result)

    q.put("kill")
    pool.close()
    pool.join()
    queue_process.join()


def handle_log_queue(q, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        while True:
            m = q.get()
            if m == "kill":
                break
            if m is not None:
                json.dump(m, f)
                f.write("\n")
                f.flush()


def get_missing_configs(configs, path_done, relevant_params):
    df_planned = pd.DataFrame.from_records(configs)
    df_done = pd.read_json(path_done, orient="records", lines=True)

    # Match on specific columns
    diff_df = df_planned.merge(df_done, on=relevant_params, how="left", indicator=True)
    missing_mask = diff_df["_merge"] == "left_only"
    result = [config for config, is_missing in zip(configs, missing_mask) if is_missing]
    # Convert back to list of configs using boolean indexing
    return result


def run(xs, ys, hparams, anom_filter=None, device="cpu", seed=42, verbose=True):
    model = None
    optimizer = None
    torch.manual_seed(seed)

    preds = []
    loss_weights = []

    data = zip(xs, ys)
    iterator = tqdm(data) if verbose else data
    for x, y in iterator:
        if model is None:
            model = get_mlp(
                in_features=x.shape[-1],
                out_features=hparams["n_classes"],
                n_hidden_units=hparams["n_hidden_units"],
                n_hidden_layers=hparams["n_hidden_layers"],
            )
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=hparams["lr"])

        # x, y = x.to(device), y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=-1)
        preds.append(pred.detach().cpu().item())

        loss = F.cross_entropy(logits, y)

        if anom_filter is not None:
            loss_weight = anom_filter(x)
            loss = loss * loss_weight
            loss_weights.append(loss_weight)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if loss_weights:
        return preds, loss_weights
    else:
        return preds


def pretrain_autoencoder(
    xs,
    n_hidden_layers,
    n_hidden_units,
    dropout,
    lr,
    epochs=5,
    seed=42,
    device="cpu",
    verbose=True,
):
    model = None
    optimizer = None
    torch.manual_seed(seed)

    for i in range(epochs):
        iterator = tqdm(xs) if verbose else xs
        for x in iterator:
            if model is None:
                model = get_autoencoder(
                    in_features=x.shape[-1],
                    dropout=dropout,
                    n_hidden_units=n_hidden_units,
                    n_hidden_layers=n_hidden_layers,
                )
                model = model.to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            x = x.to(device)

            model.train()
            x_rec = model(x)
            loss = F.l1_loss(x_rec, x)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model
