# Learning Rate Optimization in Online Deep Learning 

To reproduce the experiments: 

1. Create new anaconda environment with `conda env create -n lodl --file environment.yaml`.
2. Activate environment with `conda activate lodl`.
3. Run experiments with `run_experiments.sh`.
4. To generate tables, run notebook `notebooks/tables.ipynb`.
5. To create Figure 2 which shows accuracy and learning rate of different schedules and optimizers throughout a stream, run `notebooks/lr_plots.ipynb`.
6. To create Figure 3 which shows the accuracy achieved with our pre-tuning method, run `notebooks/pretune.ipynb`.