import json
from pathlib import Path

import numpy as np
import torch
from base import get_config_grid
from src.data.contamination import get_tuning_data
from contam_base import run
from tqdm import tqdm

if __name__ == "__main__":
    device = "cpu"
    run_name = "tune_full_insects.jsonl"
    dataset = 'Insects abrupt'
    logpath = Path(__file__).parent.parent.joinpath("reports", run_name)
    tuning_samples = 1000
    tuning_steps = 5000
    tuning_configs = get_config_grid(
        **{
            "n_hidden_units": [512, 1024],
            "n_hidden_layers": [1, 2, 3],
            "lr": [2**-i for i in range(6)],
            "n_classes": 2,
        }
    )
    xs, ys = get_tuning_data(dataset, tuning_samples=tuning_samples)
    idcs_resample = np.random.choice(tuning_samples, size=tuning_steps, replace=True)
    xs_res, ys_res = xs[idcs_resample], ys[idcs_resample]
    xs_res, ys_res = (
        torch.tensor(xs_res, dtype=torch.float),
        torch.tensor(ys_res, dtype=torch.long),
    )

    for config in tqdm(tuning_configs, desc='Running configs'):
        preds = run(xs_res, ys_res, config, device=device, verbose=False)
        result = config | {'dataset': dataset, "preds": preds, "labels": ys_res.tolist()}
        with open(logpath, "a") as f:
            f.write(json.dumps(result) + "\n")
