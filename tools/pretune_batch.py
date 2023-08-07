import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from src.models.networks import get_mlp
from src.models.schedulers import ChainedScheduler, LRLimiter
from tools.base import REPORTS_PATH, bce_with_logits, test_configs
from tools.pretune import count_parameters, configs, datasets
from torch.utils.data import DataLoader, Subset

run_name = "v3"
save_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")


def write_result(result):
    has_header = False
    if save_path.exists():
        with open(save_path, "r") as f:
            has_header = f.read(1024) != ""
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results captured since last logging step
    with open(save_path, "a") as f:
        writer = csv.DictWriter(f, result.keys())
        if not has_header:
            writer.writeheader()
        writer.writerow(result)


def tune_batch_mode(
    lr,
    batch_size,
    n_hidden_layers,
    n_hidden_units,
    n_samples,
    gamma,
    seed,
    data,
    dataset_name,
    validation_split=0.2,
    num_epochs=20,
    device="cpu",
    verbose=False,
):
    torch.manual_seed(seed)
    x_sample, y_sample = data[0]
    net = get_mlp(
        x_sample.shape[-1],
        y_sample.shape[-1],
        n_hidden_layers=n_hidden_layers,
        hidden_features=n_hidden_units,
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    if gamma < 1:
        scheduler = ChainedScheduler(
            [
                ExponentialLR(optimizer, gamma=gamma),
                LRLimiter(optimizer, min_lr=0.1),
            ]
        )
    else:
        scheduler = None
    n_params = count_parameters(net)
    n_validation = int(n_samples * validation_split)
    x, y = data.tensors
    idcs = np.arange(n_samples)
    train_idcs, val_idcs = train_test_split(
        idcs, random_state=seed, test_size=validation_split
    )
    data_train = Subset(data, train_idcs)
    data_val = Subset(data, val_idcs)

    train_loader = DataLoader(data_train, batch_size=batch_size)
    val_loader = DataLoader(data_val, batch_size=batch_size)
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        if verbose:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = net(inputs)
            loss = bce_with_logits(labels, logits).sum()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        # Validation
        net.eval()
        with torch.inference_mode():
            total_val_loss = 0.0
            total_correct = 0

            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_logits = net(val_inputs)
                val_loss = bce_with_logits(val_labels, val_logits).sum()
                total_correct += (
                    (
                        torch.argmax(val_logits, dim=-1)
                        == torch.argmax(val_labels, dim=-1)
                    )
                    .sum()
                    .item()
                )
                total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_loader)
            best_val_loss = min(best_val_loss, average_val_loss)
            average_val_accuracy = total_correct / n_validation

            # Check for early stopping
            if average_val_accuracy > best_val_accuracy:
                best_val_accuracy = average_val_accuracy
                best_epoch = epoch

            if verbose:
                print(f"Train loss: {total_loss / len(train_loader)}")
                print(f"Validation loss: {average_val_loss}")
                print(f"Validation accuracy: {average_val_accuracy}")

        result = {
            "base_lr": lr,
            "dataset": dataset_name,
            "batch_size": batch_size,
            "seed": seed,
            "n_hidden_layers": n_hidden_layers,
            "n_hidden_units": n_hidden_units,
            "n_params": n_params,
            "n_samples": n_samples,
            "gamma": gamma,
            "schedule": "Exp",
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
    write_result(result)
    return best_val_accuracy


if __name__ == "__main__":
    test_configs(
        test_func=tune_batch_mode,
        dataset_names=datasets,
        configs=configs,
        debug=False,
    )
