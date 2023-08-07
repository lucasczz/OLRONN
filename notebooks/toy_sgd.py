from typing import Iterable, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.datasets import make_classification
from scipy.optimize import differential_evolution
import pandas as pd


class DoG:
    def __init__(self, grad_fn, lr=0.8, r_c=1, momentum=None, weight_decay=None):
        self.gamma = lr
        self.t = 1
        self.grad_fn = grad_fn
        self.max_weight_diff_norm = np.zeros(2)
        self.sq_norms_sum = np.zeros(2)
        self.avg_w = None
        self.lr = lr
        self.r_c = r_c
        self.n_observed = 0
        self.init_weight = None

    def step(self, w, x, y):
        grad = self.grad_fn(w, x, y)
        sq_norm = np.dot(grad, grad)
        self.sq_norms_sum += sq_norm
        if self.init_weight is None:
            self.init_weight = w
            self.avg_w = w
            eta = 1 / np.linalg.norm(grad) * self.r_c
        else:
            weight_diff_norm = np.linalg.norm(w - self.init_weight)
            self.max_weight_diff_norm = np.maximum(
                weight_diff_norm, self.max_weight_diff_norm
            )
            eta = self.max_weight_diff_norm / np.sqrt(self.sq_norms_sum)
        w_new = w - eta * grad
        self.avg_w = (self.t + 1) / (self.t + self.gamma) * self.avg_w + (
            self.gamma + 1
        ) / (self.t + self.gamma) * w_new
        self.t += 1
        return w - self.avg_w


class COCOB:
    def __init__(
        self, grad_fn, lr=100.0, eps=1e-8, gamma=None, momentum=None, weight_decay=None
    ):
        self.grad_fn = grad_fn
        self.alpha = lr
        self.lr = lr
        self.eps = eps
        self.eps = eps
        self.t = 0
        self.initial_weight = None
        self.reward = np.zeros(2)
        self.bet = np.zeros(2)
        self.grads_sum = np.zeros(2)
        self.absolute_grads_sum = np.zeros(2)
        self.max_observed_scale = self.eps * np.ones(2)

        self.n_observed = 0

    def step(self, w, x, y):
        grad = self.grad_fn(w, x, y)
        if self.initial_weight is None:
            self.initial_weight = w

        abs_grad = np.abs(grad)
        self.max_observed_scale = np.maximum(self.max_observed_scale, abs_grad)
        self.absolute_grads_sum += abs_grad
        self.grads_sum += grad

        win_amount = self.bet * grad
        self.reward = np.maximum(self.reward + win_amount, np.zeros_like(self.reward))
        bet_fraction = self.grads_sum / (
            self.max_observed_scale
            * (
                np.maximum(
                    self.absolute_grads_sum + self.max_observed_scale,
                    self.alpha * self.max_observed_scale,
                )
            )
        )
        bet = bet_fraction * (self.max_observed_scale + self.reward)

        update = bet - self.bet
        self.bet = bet
        self.t += 1

        self.n_observed += x.shape[0]
        return update


class Adam:
    def __init__(
        self,
        grad_fn,
        lr=0.4,
        lr_schedule_fn=None,
        gamma=None,
        momentum=None,
        beta1=0.9,
        beta2=0.99,
        eps=1e-8,
        weight_decay=0.01,
        fallback_state=None,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr = lr
        self.n_observed = 0
        self.lr_schedule_fn = lr_schedule_fn
        self.grad_fn = grad_fn
        self.fallback_state = fallback_state

    def step(self, w, x, y):
        grad = self.grad_fn(w, x, y) + self.weight_decay * w
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_ = self.m / (1 - self.beta1 ** (self.t + 2))
        v_ = self.v / (1 - self.beta2 ** (self.t + 2))
        update = self.lr * m_ / (v_**0.5 + self.eps)
        if self.lr_schedule_fn:
            self.lr = self.lr_schedule_fn(self.lr, self.n_observed)
        self.n_observed += x.shape[0]
        self.t += 1
        return update

    def on_concept_change(self):
        if self.fallback_state is not None:
            for key, item in self.fallback_state.items():
                setattr(self, key, item)


class SGD:
    def __init__(
        self,
        grad_fn,
        lr=5e-3,
        momentum=0.1,
        weight_decay=0.01,
        fallback_state=None,
    ) -> None:
        self.lr = lr
        self.grad_fn = grad_fn
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.n_observed = 0
        self.fallback_state = fallback_state

    def step(self, w, x, y):
        grad = self.grad_fn(w, x, y) + self.weight_decay * w
        if self.t > 0:
            self.m = self.momentum * self.m + grad
        else:
            self.m = grad
        self.t += 1
        self.n_observed += x.shape[0]
        update = self.m * self.lr
        return update

    def on_concept_change(self):
        if self.fallback_state is not None:
            for key, item in self.fallback_state.items():
                setattr(self, key, item)


def get_trajectory(w0, optim, data, lr_scheduler, reset_at=None):
    w_t = w0
    trajectory = [w_t]
    for x, y in data:
        update = optim.step(w_t, x, y)
        w_t = w_t - update
        lr_scheduler.step()
        trajectory.append(w_t)

    return np.stack(trajectory).T


class ExponentialLR:
    def __init__(self, optimizer, gamma, reset_at=None) -> None:
        self.optimizer = optimizer
        self.gamma = gamma
        self.reset_at = reset_at
        self.base_lr = optimizer.lr

    def step(self):
        if self.optimizer.n_observed == self.reset_at:
            if isinstance(self.optimizer, Adam):
                self.optimizer.m = np.zeros(2)
                self.optimizer.v = np.zeros(2)
                self.optimizer.t = 0
            self.optimizer.lr = self.base_lr
        else:
            self.optimizer.lr *= self.gamma


def bce_with_logits(w, x, y, reduction="sum"):
    logits = w @ x.T
    y_hat = sigmoid(logits)
    raw_loss = -(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
    if reduction == "sum":
        return np.sum(raw_loss, axis=-1)
    elif reduction == "mean":
        return np.mean(raw_loss, axis=-1)


# x = (n_samples x 2)
# y_hat - y = (n_samples)
# x.T @ (y_hat - y) = (2 x n_samples) @ (n_samples) = (2)
def bce_grad(w, x, y):
    logits = w @ x.T
    y_hat = sigmoid(logits)
    grad = x.T @ (y_hat - y)
    return grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_classification_data(n_samples=20, seed=0):
    # Function for generating data

    data = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=2,
        hypercube=True,
        class_sep=0.5,
        shuffle=True,
        random_state=seed,
    )
    x, y = data
    w = differential_evolution(
        bce_with_logits, bounds=[(-10, 10), (-10, 10)], args=(x, y), tol=1e-6
    ).x
    return x, y, w


def split_into_batches(*arrays, batch_size):
    return [np.array_split(array, len(array) // batch_size) for array in arrays]


def run(
    x,
    y,
    n_train_samples,
    batch_size,
    optimizer="SGD",
    gamma=1,
    reset_at=None,
    lr=0.4,
    momentum=0,
    w0=-np.ones(2),
):
    batches_x, batches_y = split_into_batches(
        x[:n_train_samples], y[:n_train_samples], batch_size=batch_size
    )
    batches = list(zip(batches_x, batches_y))
    optim_fn = {"SGD": SGD, "Adam": Adam, "COCOB": COCOB, "DoG": DoG}.get(optimizer)
    optim = optim_fn(
        grad_fn=bce_grad,
        lr=lr,
        momentum=momentum,
        weight_decay=0,
    )
    lr_scheduler = ExponentialLR(optim, gamma=gamma, reset_at=reset_at)
    return get_trajectory(w0=w0, optim=optim, data=batches, lr_scheduler=lr_scheduler)


def tumbling_mean(arr, window_size):
    means = []
    measuring_points = []
    for start in range(0, len(arr) - window_size, window_size):
        end = start + window_size
        window = arr[start:end]
        mean = np.mean(window)
        means.append(mean)
        measuring_points.append(end)
    return np.array(means).flatten(), np.array(measuring_points)


DEFAULT_TRAJECTORY_PROPS = {
    "column": 0,
    "color": "black",
    "cmap": "viridis",
    "linestyle": "solid",
    "concept": 0,
    "marker_interval": 8,
    "marker": "o",
    "label": "",
    "marker_cmap": "concept",
}
DEFAULT_CONCEPT_PROPS = {
    "color": None,
    "linestyles": "solid",
    "column": 0,
    "cmap": None,
}


def move_arrow_destinations(destinations, origins, margin):
    # Calculate direction vectors
    directions = destinations - origins

    # Calculate magnitudes of direction vectors
    magnitudes = np.linalg.norm(directions, axis=1)

    # Normalize direction vectors
    normalized_directions = directions / magnitudes[:, np.newaxis]

    # Define the margin by which to move destinations
    margin = 0.1

    # Update destinations by moving them towards origins by the margin
    return destinations - normalized_directions * margin


def plot_loss_trajectory(
    trajectories: list,
    concepts: dict,
    gridspec=None,
    contour_levels=15,
    markersize=60,
    figsize=[5, 4],
    spacing=(0.2, 0.2),
    sharex=False,
    sharey=False,
    axs=None,
    xlims=None,
    ylims=None,
):
    for i in range(len(trajectories)):
        trajectories[i] = DEFAULT_TRAJECTORY_PROPS | trajectories[i]

    for i in range(len(concepts)):
        concepts[i] = DEFAULT_CONCEPT_PROPS | concepts[i]

    trajectories = pd.DataFrame(trajectories)
    if xlims is None:
        xlims = {col: np.array((1e8, -1e8)) for col in trajectories["column"].unique()}
    else:
        xlims = {col: xlims for col in trajectories["column"].unique()}
    if ylims is None:
        ylims = {col: np.array((1e8, -1e8)) for col in trajectories["column"].unique()}
    else:
        ylims = {col: ylims for col in trajectories["column"].unique()}

    for _, trajectory in trajectories.iterrows():
        x, y = trajectory["points"]
        col_xlims = xlims[trajectory["column"]]
        col_ylims = ylims[trajectory["column"]]
        col_xlims[0] = min(col_xlims[0], x.min() - spacing[0])
        col_xlims[1] = max(col_xlims[1], x.max() + spacing[1])
        col_ylims[0] = min(col_ylims[0], y.min() - spacing[0])
        col_ylims[1] = max(col_ylims[1], y.max() + spacing[1])

    n_columns = trajectories["column"].nunique()
    if axs is None:
        fig, axs = plt.subplots(
            ncols=n_columns,
            gridspec_kw=gridspec,
            figsize=(figsize[0] * n_columns, figsize[1]),
            sharey=sharey,
            sharex=sharex,
        )

    contours = {}
    for column, column_trajectories in trajectories.groupby("column"):
        ax = axs[column] if isinstance(axs, np.ndarray) else axs
        ax.set_xlabel(r"$\theta_1$")
        # if column == 0:
        ax.set_ylabel(r"$\theta_2$")

        for concept_idx in column_trajectories["concept"].unique():
            concept_props = concepts[concept_idx]
            # create a grid of x and y values
            w1 = np.linspace(*xlims[column], 200)
            w2 = np.linspace(*ylims[column], 200)
            # create a meshgrid
            w1_grid, w2_grid = np.meshgrid(w1, w2)
            w_grid = np.stack((w1_grid, w2_grid), axis=-1)
            loss_surface = bce_with_logits(
                w_grid, concept_props["x"], concept_props["y"], reduction="mean"
            )
            contours[concept_idx] = ax.contour(
                w1,
                w2,
                loss_surface,
                levels=contour_levels,
                colors=concept_props["color"],
                cmap=concept_props["cmap"],
                linestyles=concept_props["linestyles"],
            )
        for _, trajectory in column_trajectories.iterrows():
            concept = concepts[trajectory["concept"]]
            points = trajectory["points"]
            losses = bce_with_logits(
                trajectory["points"].T, concept["x"], concept["y"], reduction="mean"
            )
            marker_losses, marker_indices = tumbling_mean(
                losses, trajectory["marker_interval"]
            )
            marker_points = points[:, marker_indices]
            arrow_origins = points[:, marker_indices - 1]
            arrow_destinations = marker_points
            # arrow_destinations = move_arrow_destinations(
            #     marker_points, arrow_origins, 0.001
            # )
            color = (
                concept["color"]
                if trajectory["color"] == "concept"
                else trajectory["color"]
            )
            marker_cmap = (
                contours[trajectory["concept"]].cmap
                if trajectory["marker_cmap"] == "concept"
                else trajectory["marker_cmap"]
            )
            ax.plot(
                *points,
                color=color,
                linestyle=trajectory["linestyle"],
                label=trajectory["label"],
            )

            if "origin_color" in concept:
                ax.scatter(
                    points[0, 0],
                    points[1, 0],
                    markersize,
                    marker=trajectory["marker"],
                    c=concept["origin_color"],
                    edgecolors=color,
                    zorder=3,
                )
            if "arrow_size" in trajectory:
                for origin, dest in zip(arrow_origins.T, arrow_destinations.T):
                    arrow = patches.FancyArrowPatch(
                        origin,
                        dest,
                        color=color,
                        arrowstyle="-|>",
                        mutation_scale=trajectory["arrow_size"],
                        zorder=3,
                        linewidth=0,
                        # linestyle=trajectory['lin']
                    )
                    ax.add_patch(arrow)
            if trajectory["marker"] != "none":
                ax.scatter(
                    *marker_points,
                    markersize,
                    c=marker_losses,
                    marker=trajectory["marker"],
                    zorder=3,
                    edgecolors=color,
                    cmap=marker_cmap,
                    norm=contours[trajectory["concept"]].norm,
                )

    return axs, contours


def get_best_lr(
    lrs,
    x,
    y,
    n_train_samples,
    optimizer="SGD",
    batch_size=2,
    gamma=1,
    reset_at=None,
    momentum=0,
    w0=-np.ones(2),
):
    best_loss = 1e8
    for lr in lrs:
        trajectory = run(
            x,
            y,
            n_train_samples=n_train_samples,
            optimizer=optimizer,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            reset_at=reset_at,
            momentum=momentum,
            w0=w0,
        )
        mean_loss = bce_with_logits(trajectory.T, x, y).mean()
        if mean_loss <= best_loss:
            best_loss = mean_loss
            best_lr = lr
            best_trajectory = trajectory
    return best_lr, best_trajectory


def create_drift_data(concepts):
    xs, ys = [], []
    for concept in concepts:
        x, y, _ = generate_classification_data(
            n_samples=concept["n_samples"], seed=concept["seed"]
        )
        xs.append(x)
        ys.append(y)
    return xs, ys


def get_trajectory_comparison(
    concepts: List[dict],
    trajectories: List[dict],
    n_train_samples: int,
    w0=-np.ones(2),
    gridspec: dict = None,
    figsize: List = [5, 4],
    contour_levels: int = 10,
    markersize: int = 100,
    axs=None,
    xlims=None,
    ylims=None,
):
    xs, ys = create_drift_data(concepts)
    for x, y, concept in zip(xs, ys, concepts):
        concept["x"] = x
        concept["y"] = y
    x, y = np.concatenate(xs), np.concatenate(ys)

    sub_trajectories = []
    for trajectory in trajectories:
        batch_size = trajectory.get("batch_size", 4)
        gamma = trajectory.get("gamma", 1)
        reset_at = trajectory.get("reset_at", None)
        optimizer = trajectory.get("optimizer", "SGD")
        momentum = trajectory.get("momentum", 0)
        lr = trajectory.get("lr", [2**-i for i in range(-1, 4)])
        if isinstance(lr, Iterable):
            _, points = get_best_lr(
                lrs=lr,
                x=x,
                y=y,
                optimizer=optimizer,
                n_train_samples=n_train_samples,
                batch_size=batch_size,
                gamma=gamma,
                momentum=momentum,
                reset_at=reset_at,
                w0=w0,
            )
        else:
            points = run(
                x=x,
                y=y,
                n_train_samples=n_train_samples,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                reset_at=reset_at,
                lr=lr,
                w0=w0,
            )
        start = 0
        for idx, concept in enumerate(concepts):
            end = start + concept["n_samples"] // batch_size

            sub_trajectory = concept | trajectory
            sub_trajectory.update(
                {
                    "concept": idx,
                    "points": points[:, start : end + 1],
                }
            )
            sub_trajectories.append(sub_trajectory)
            start = end

    return plot_loss_trajectory(
        sub_trajectories,
        concepts,
        gridspec=gridspec,
        figsize=figsize,
        contour_levels=contour_levels,
        markersize=markersize,
        axs=axs,
        xlims=xlims,
        ylims=ylims,
    )
