# Optimizing the Learning Rate for Online Training of Neural Networks 

## Reproduction

To reproduce the experiments: 

1. Create new anaconda environment with `conda env create --file environment.yaml`.
2. Activate environment with `conda activate olronn`.
3. Run experiments with `run_experiments.sh`.
4. To generate tables, run notebook `notebooks/tables.ipynb`.
5. To create Figure 2 which shows accuracy and learning rate of different schedules and optimizers throughout a stream, run `notebooks/lr_plots.ipynb`.
6. To create Figure 3 which shows the accuracy achieved with our pre-tuning method, run `notebooks/pretune.ipynb`.

## Hyperparameter Settings


### Parameter Notation as used in Paper
| Parameter                        | Symbol       |
| -------------------------------- | ------------ |
| Learning Rate                    | $\eta$       |
| Decay Factor                     | $\gamma$     |
| Drift Detection Confidence Level | $\delta$     |
| Steps Between LR Cycles/Updates  | $s$          |
| Relative LR at Midpoint of Cycle | $\hat{\eta}$ |

### Search spaces for learning rate
| Optimizer | Learning Rate Search Space                |
| --------- | ----------------------------------------- |
| SGD       | ${2^1, 2^0, ..., 2^{-8}}$                 |
| Adam      | ${2^{-3}, 2^{-4}, ..., 2^{-12}}$          |
| AdaGrad   | ${2^1, 2^0, ..., 2^{-8}}$                 |
| WNGrad    | ${10^{1.25}, 10^{0.75}, ..., 10^{-7.75}}$ |
| HD        | ${2^{-3}, 2^{-4}, ..., 2^{-12}}$          |
| COCOB     | $100$                                     |
| DoG       | $1$                                       |
| D-Adapt   | $1$                                       |
| Mechanic  | $0.01$                                    |

### Other parameter settings
| Schedule    | Values                                  |
| ----------- | --------------------------------------- |
| Exponential | $\gamma = 1 - 2^{-13}$                  |
| Exp. Reset  | $\gamma = 1 - 2^{-12}, \delta = 0.0001$ |
| Step        | $\gamma = 0.75, s = 2000$               |
| Cyclic      | $\hat{\eta} = 0.25, s = 8000$           |
