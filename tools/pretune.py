from pathlib import Path
from base import (
    DATASET_NAMES,
    DATASETS_REAL,
    DATASETS_SYNTH,
    REPORTS_PATH,
    SEEDS,
    get_config_grid,
    run_configs,
    tune_batch_mode,
    tune_prequential,
    zip_csv,
)

# Set up logging path
log_path = REPORTS_PATH.joinpath(Path(__file__).stem)

datasets = DATASET_NAMES

lr = [2**-i for i in range(6)]
n_hidden_layers = [1, 3]
n_hidden_units = [64, 128]
n_samples = [500, 1000]
gamma = [1 - 2**-i for i in range(12, 15)] + [1]

configs = get_config_grid(
    base_lr=lr,
    n_hidden_layers=n_hidden_layers,
    n_hidden_units=n_hidden_units,
    n_samples=n_samples,
    gamma=gamma,
    seed=SEEDS,
    optimizer="SGD",
)

if __name__ == "__main__":
    preq_path = log_path.joinpath("prequential.csv")
    run_configs(
        run_fn=tune_prequential,
        dataset_names=datasets,
        configs=configs,
        debug=False,
        log_path=preq_path,
    )
    zip_csv(preq_path)
    batch_path = log_path.joinpath("batch_mode.csv")
    run_configs(
        run_fn=tune_batch_mode,
        dataset_names=datasets,
        configs=configs,
        debug=False,
        log_path=batch_path,
    )
    zip_csv(batch_path)
