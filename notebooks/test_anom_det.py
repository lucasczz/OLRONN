import numpy as np
from tqdm import tqdm

from river.datasets import synth
from src.models.networks import get_mlp
from src.data import get_dataset, river2np_dataset
from torchvision.datasets import MNIST
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from torchvision.datasets import FashionMNIST


anom_type = "ood"
p_anom = 0.05
anom_class = 3
device = "cuda:0"


np.random.seed(42)
torch.manual_seed(42)

# mnist_data = MNIST(root="./data", train=True, download=True)
# xs, ys = mnist_data.data, mnist_data.targets
# xs = xs / 255
# xs, ys = xs.numpy(), ys.numpy()
# xs = xs.reshape(len(ys), -1)

# fmnist_data = FashionMNIST(root="./data", train=True, download=True)
# xs_ood = fmnist_data.data
# xs_ood = xs_ood / 255
# xs_ood = xs_ood.numpy()
# xs_ood = xs_ood.reshape(len(ys), -1)

xs, ys = get_dataset("Covertype")

ood_stream = synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=10, n_features=xs.shape[-1])

# x0 = xs[ys != anom_class]
# y0 = ys[ys != anom_class]
# x1 = xs[ys == anom_class]

x0 = xs
y0 = ys
x1 = xs_ood

n_train = round(len(x0) * 0.2)

n_normal = len(x0)
n_anom = round(n_normal * p_anom / (1 - p_anom))
anom_insert_idcs = np.sort(
    np.random.choice(np.arange(n_normal), size=n_anom, replace=False)
)[::-1]
y_unique = np.unique(y0)


is_anom = np.zeros(len(x0)).astype("bool")
if anom_type == "label_flip":
    x_contam, y_contam = x0.copy(), y0.copy()
    # y_contam[anom_insert_idcs] = np.random.choice(y_unique, size=n_anom)
    y_contam[anom_insert_idcs] = anom_class
    is_anom[anom_insert_idcs] = True
elif anom_type == 'feature_swap':
    swap_ratio = .1
    n_swap = round(swap_ratio * x0.shape[-1])
    swap_idcs = np.random

    x_contam, y_contam = x0.copy(), y0.copy()
    # y_contam[anom_insert_idcs] = np.random.choice(y_unique, size=n_anom)
    y_contam[anom_insert_idcs] = anom_class
    is_anom[anom_insert_idcs] = True
else:
    anom_idcs = np.random.choice(len(x1), size=n_anom, replace=False)
    anoms = x1[anom_idcs]
    x_contam, y_contam, is_anom = x0.tolist(), y0.tolist(), is_anom.tolist()
    for idx, anom in zip(anom_insert_idcs, anoms):
        x_contam.insert(idx, anom)
        y_contam.insert(idx, anom_class)
        # y_contam.insert(idx, np.random.choice(y_unique))

        is_anom.insert(idx, True)

    x_contam = np.array(x_contam)
    y_contam = np.array(y_contam)


label_enc = LabelEncoder()
y_contam = label_enc.fit_transform(y_contam)
is_anom = np.array(is_anom)

data_contam = TensorDataset(
    torch.tensor(x_contam, dtype=torch.float),
    torch.tensor(y_contam, dtype=torch.long),
    torch.tensor(is_anom),
)
loader_contam = DataLoader(data_contam, batch_size=1, shuffle=False)

torch.manual_seed(42)

accuracies = []
for skip_anomalies in [True, False]:
    model = get_mlp(
        in_features=x_contam.shape[-1],
        out_features=y_contam.max() + 1,
        n_hidden_units=256,
        n_hidden_layers=1,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    all_preds = []
    for x, y, is_anom_i in tqdm(loader_contam):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=-1)
        all_preds.append(pred.detach().cpu().item())

        if skip_anomalies and is_anom_i:
            continue

        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    all_preds = np.array(all_preds)

    accuracies.append((all_preds == y_contam)[~is_anom].mean())
print(accuracies[0] - accuracies[1])