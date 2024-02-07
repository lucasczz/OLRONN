from pathlib import Path

from tools.base import (
    DATASETS_SYNTH,
    get_config_grid,
    run_configs,
    REPORTS_PATH,
    SEEDS,
    LRS,
    DATASET_NAMES,
    zip_csv,
)

# Set up logging path
run_name = "v1"
log_path = REPORTS_PATH.joinpath(Path(__file__).stem, f"{run_name}.csv")

# Set up configs
batch_sizes = [2**i for i in range(5)]
datasets = [f"RBF incr._{i*0.001}" for i in range(2)] + DATASET_NAMES

configs = get_config_grid(batch_size=batch_sizes, base_lr=LRS, seed=SEEDS)


if __name__ == "__main__":
    run_configs(dataset_names=DATASETS_SYNTH, configs=configs, debug=False, log_path=log_path)
    zip_csv(log_path)
