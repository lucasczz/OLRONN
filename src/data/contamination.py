import numpy as np
from river.datasets import synth
from sklearn.preprocessing import LabelEncoder

from src.data.datasets import get_dataset, river2np_dataset

NORMAL_CLASSES = {
    "Covertype": [1, 2],
    "Insects abrupt": ["ae-albopictus-male", "ae-albopictus-female"],
}
ANOMALY_CLASSES = {
    "Covertype": [4],
    "Rotated MNIST": [4],
    "Insects abrupt": ["ae-aegypti-male", "ae-aegypti-female"],
}


def get_anom_idcs(total_len, len_anomaly=6, p_anomaly=0.05):
    anom_insert_idcs = []
    current_is_anom = False

    adj_p_anomaly = p_anomaly / (1 - p_anomaly)
    p_anom_normal = 1 / len_anomaly
    p_normal_anom = adj_p_anomaly / (len_anomaly * (1 - adj_p_anomaly))

    exit_probas = {False: p_normal_anom, True: p_anom_normal}
    for idx, p in enumerate(np.random.rand(total_len)):
        if p < exit_probas[current_is_anom]:
            current_is_anom = not current_is_anom
        if current_is_anom:
            anom_insert_idcs.append(idx)
    return anom_insert_idcs[::-1]


def get_tuning_data(
    dataset,
    tuning_samples=2000,
):
    xs_all, ys_all = get_dataset(dataset)

    normal_classes = NORMAL_CLASSES.get(dataset, None)
    anomaly_classes = ANOMALY_CLASSES.get(dataset, None)

    if normal_classes is not None:
        id_mask = np.isin(ys_all, normal_classes)
    else:
        id_mask = ys_all != anomaly_classes
    xs, ys = xs_all[id_mask], ys_all[id_mask]

    xs, ys = xs[:tuning_samples], ys[:tuning_samples]
    ys = LabelEncoder().fit_transform(ys)

    return xs, ys


def get_contaminated_stream(
    dataset,
    anomaly_type,
    p_anomaly,
    len_anomaly,
    tuning_samples=2000,
    anom_label="random",
    seed=42,
):
    np.random.seed(seed)

    xs_all, ys_all = get_dataset(dataset)

    normal_classes = NORMAL_CLASSES.get(dataset, None)
    anomaly_classes = ANOMALY_CLASSES.get(dataset, None)
    if normal_classes is not None:
        id_mask = np.isin(ys_all, normal_classes)
    else:
        id_mask = np.isin(ys_all, anomaly_classes)
    xs, ys = xs_all[id_mask], ys_all[id_mask]

    xs, ys = xs[tuning_samples:], ys[tuning_samples:]

    anom_insert_idcs = get_anom_idcs(len(xs), len_anomaly, p_anomaly)

    n_anom = len(anom_insert_idcs)

    y_unique = np.unique(ys)

    is_anom = np.zeros(len(xs)).astype("bool")
    if anomaly_type == "ood_sample":
        ood_stream = synth.RandomRBF(
            seed_model=42, seed_sample=42, n_classes=10, n_features=xs.shape[-1]
        ).take(n_anom)
        xs_ood, _ = river2np_dataset(ood_stream)
    elif anomaly_type == "ood_class":
        xs_ood = xs_all[np.isin(ys_all, anomaly_classes)]

    if anomaly_type == "label_flip":
        x_contam, y_contam = xs.copy(), ys.copy()
        y_contam[anom_insert_idcs] = (
            np.random.choice(y_unique, size=n_anom)
            if anom_label == "random"
            else anom_label
        )
        is_anom[anom_insert_idcs] = True
    elif anomaly_type == "feature_swap":
        x_contam, y_contam = xs.copy(), ys.copy()
        is_anom[anom_insert_idcs] = True
        n_features = xs.shape[1]
        # Precompute random swaps for all indices to avoid repeated np.random calls in loop
        num_swaps = np.random.randint(
            1, n_features // 2 , size=len(anom_insert_idcs)
        ) * 2
        swap_features = [
            np.random.choice(n_features, size=num_swap, replace=False)
            for num_swap in num_swaps
        ]
        for idx, features in zip(anom_insert_idcs, swap_features):
            num_swap = len(features)
            features_from = features[: num_swap // 2]
            features_to = features[num_swap // 2 :]
            swap_vals = x_contam[idx, features_from].copy()
            x_contam[idx, features_from] = x_contam[idx, features_to]
            x_contam[idx, features_to] = swap_vals
    elif anomaly_type == "gaussian_noise":
        x_contam, y_contam = xs.copy(), ys.copy()

        noise = (
            np.random.normal(
            loc=0.0,
            scale=np.std(x_contam, axis=0),
            size=(len(anom_insert_idcs), x_contam.shape[1]),
            )
            * np.random.rand(len(anom_insert_idcs))[:, None]
        )
        x_contam[anom_insert_idcs] += noise

        # Clamp values to original min/max per feature
        min_vals = np.min(xs, axis=0)
        max_vals = np.max(xs, axis=0)
        x_contam[anom_insert_idcs] = np.clip(
            x_contam[anom_insert_idcs], min_vals, max_vals
        )

        is_anom[anom_insert_idcs] = True

    else:
        anom_idcs = np.random.choice(len(xs_ood), size=n_anom, replace=True)
        anoms = xs_ood[anom_idcs]
        x_contam, y_contam, is_anom = xs.tolist(), ys.tolist(), is_anom.tolist()
        for idx, anom in zip(anom_insert_idcs, anoms):
            x_contam.insert(idx, anom)
            new_label = (
                np.random.choice(y_unique) if anom_label == "random" else anom_label
            )
            y_contam.insert(idx, new_label)

            is_anom.insert(idx, True)

        x_contam = np.array(x_contam)
        y_contam = np.array(y_contam)

    label_enc = LabelEncoder()
    y_contam = label_enc.fit_transform(y_contam)
    is_anom = np.array(is_anom)

    return x_contam, y_contam, is_anom
