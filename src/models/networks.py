from torch import nn


def get_mlp(in_features=20, out_features=5, hidden_features=None, n_hidden_layers=1, activation=nn.ReLU):
    if hidden_features is None:
        hidden_features = in_features
    return nn.Sequential(
        *[nn.Linear(in_features, hidden_features), activation()] * min(n_hidden_layers, 1),
        *[nn.Linear(hidden_features, hidden_features), activation()]
        * (n_hidden_layers - 1),
        nn.Linear(hidden_features, out_features),
    )
