from sklearn.datasets import fetch_covtype, fetch_openml
from river.datasets import synth, Insects
from river.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from functools import partial
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import minmax_scale


def river2np_dataset(river_data):
    x = np.array([list(sample[0].values()) for sample in river_data])
    y = np.array([sample[1] for sample in river_data])
    return x, y


def fetch_rbf_gradual(
    n_samples=100_000,
    drift_width=5000,
    n_classes=5,
    n_features=20,
    n_centroids=20,
    seed=42,
):
    np.random.seed(seed)
    before = synth.RandomRBF(
        seed_sample=seed,
        seed_model=seed,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
    )
    after = synth.RandomRBF(
        seed_sample=seed + 1,
        seed_model=seed + 1,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
    )
    if drift_width < 100:
        n_after = n_samples // 2
        before = list(before.take(n_samples - n_after))
        after = list(after.take(n_after))
        x, y = river2np_dataset(before + after)
    else:
        position = n_samples // 2
        stream = synth.ConceptDriftStream(
            before, after, position=position, width=drift_width, seed=seed
        )
        x, y = river2np_dataset(list(stream.take(n_samples)))
    x = minmax_scale(x)
    return x, y


def fetch_rbf_incremental(
    n_samples=100_000,
    drift_speed=0.001,
    n_classes=5,
    n_features=20,
    n_centroids=20,
    seed=42,
):
    np.random.seed(seed)
    stream = synth.RandomRBFDrift(
        seed,
        seed,
        n_classes=n_classes,
        n_features=n_features,
        n_centroids=n_centroids,
        change_speed=drift_speed,
        n_drift_centroids=n_centroids,
    )

    return river2np_dataset(list(stream.take(n_samples)))


def fetch_electricity():
    x, y = fetch_openml(data_id=151, return_X_y=True, as_frame=True, parser="auto")
    x = pd.get_dummies(x, dtype=float)
    return x.to_numpy(), y


def fetch_electricity_change():
    x, y = fetch_openml(data_id=151, return_X_y=True, as_frame=True, parser="auto")
    x = pd.get_dummies(x, dtype=float)

    y_change = [1] + [y[i + 1] != y[i] for i in range(len(y) - 1)]
    return x.to_numpy(), np.array(y_change, dtype=float)


def fetch_insects(variant):
    stream = Insects(variant)
    return river2np_dataset(stream)


def fetch_airlines():
    x, y = fetch_openml(data_id=1169, return_X_y=True, as_frame=True, parser="auto")
    x = pd.get_dummies(x)
    return x.to_numpy(), y.to_numpy()


def fetch_sea(n_samples=20000, change_points=(5000, 15000), seed=42):
    feats = np.random.uniform(high=10, size=(n_samples, 3))
    feat_sum = feats[:, 0] + feats[:, 1]
    thresholds = np.ones_like(feat_sum) * 8
    thresholds[change_points[0] : change_points[1]] += 1
    labels = (feat_sum > thresholds).astype(int)
    return feats, labels


def fetch_agrawal(seed=42):
    gen = synth.ConceptDriftStream(
        synth.Agrawal(classification_function=0, seed=seed, perturbation=0.1),
        synth.Agrawal(classification_function=1, seed=seed + 1, perturbation=0.1),
        position=50000,
        width=10000,
    )
    scaler = MinMaxScaler()
    stream = list(gen.take(100_000))
    x = pd.DataFrame([scaler.learn_one(xi).transform_one(xi) for xi, yi in stream])
    y = [yi for xi, yi in stream]
    x = pd.get_dummies(x, dtype=float)
    return x.to_numpy(dtype=float), np.array(y)


def fetch_led(seed=42):
    np.random.seed(seed)
    gen = synth.LED(noise_percentage=0.1, seed=seed, irrelevant_features=True)
    stream = list(gen.take(100_000))
    x = pd.DataFrame([xi for xi, yi in stream]).to_numpy(dtype=float)
    y = [yi for xi, yi in stream]
    relevant_cols_to_swap = np.random.choice(np.arange(0, 7), size=5, replace=False)
    irrelevant_cols_to_swap = np.random.choice(np.arange(7, 24), size=5, replace=False)
    x_drift = x.copy()
    swap_slice = slice(25000, 75000)
    x_drift[swap_slice, relevant_cols_to_swap] = x[swap_slice, irrelevant_cols_to_swap]
    x_drift[swap_slice, irrelevant_cols_to_swap] = x[swap_slice, relevant_cols_to_swap]
    return x_drift, np.array(y)


def fetch_rotated_mnist(
    n_experiences=4,
    rotations=[90, -45, 45, 160],
    rotation_range=(-180, 180),
    n_samples=100_000,
    seed=0,
):
    # Download MNIST dataset
    mnist_data = MNIST(root="./data", train=True, download=True)
    rng = np.random.RandomState(seed)
    xs = []
    ys = []
    if rotations is None:
        rotations = rng.randint(*rotation_range, size=n_experiences)

    n_samples_experience = n_samples // len(rotations)

    for rotation_angle in rotations:
        # Transformation to apply rotation
        transform = transforms.RandomRotation(degrees=(rotation_angle, rotation_angle))
        x = transform(mnist_data.data) / 255
        y = mnist_data.targets
        idx = rng.permutation(np.arange(len(y)))
        xs.append(x[idx].view(len(x), -1)[:n_samples_experience])
        ys.append(y[idx][:n_samples_experience])

    return np.concatenate(xs), np.concatenate(ys)


def get_dataset(name, seed=42):
    if name in DATASETS:
        return DATASETS[name]()
    elif name.startswith("RBF gradual"):
        drift_width = name.split("_")[-1]
        if drift_width[-1] == "k":
            drift_width = int(drift_width[:-1]) * 1000
        else:
            drift_width = int(drift_width)

        return fetch_rbf_gradual(
            n_samples=20000,
            drift_width=drift_width,
            n_classes=3,
            n_features=20,
            n_centroids=9,
            seed=seed,
        )
    elif name.startswith("RBF incr."):
        drift_speed = float(name.split("_")[-1])
        return fetch_rbf_incremental(
            drift_speed=drift_speed,
            n_classes=3,
            n_features=20,
            n_centroids=9,
            seed=seed,
        )


def fetch_covtype_subsample(n_samples=100_000):
    x, y = fetch_covtype(return_X_y=True)
    x = minmax_scale(x)
    return x[:n_samples], y[:n_samples]


DATASETS = {
    "RBF abrupt": partial(fetch_rbf_gradual, drift_width=0),
    "RBF gradual": partial(fetch_rbf_gradual, drift_width=5000),
    "RBF static": partial(fetch_rbf_incremental, drift_speed=0),
    "RBF incr.": partial(fetch_rbf_incremental, drift_speed=0.002),
    "Agrawal": fetch_agrawal,
    "LED": fetch_led,
    "Insects abrupt": partial(fetch_insects, "abrupt_balanced"),
    "Insects gradual": partial(fetch_insects, "gradual_balanced"),
    "Insects incr.": partial(fetch_insects, "incremental_balanced"),
    "Electricity": fetch_electricity,  # Instances: 45312,   Attributes: 14
    "Electricity change": fetch_electricity_change,  # Instances: 45312,   Attributes: 14
    "Covertype": partial(fetch_covtype_subsample),
    "SEA": fetch_sea,
    "Rotated MNIST": fetch_rotated_mnist,
}

if __name__ == "__main__":
    dataset = fetch_rotated_mnist(4)
    print(dataset[0])
