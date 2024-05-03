from torch import nn


def get_mlp(
    in_features=20,
    out_features=5,
    n_hidden_units=None,
    n_hidden_layers=1,
    activation=nn.ReLU,
):
    if n_hidden_units is None:
        n_hidden_units = in_features
    return nn.Sequential(
        *[nn.Linear(in_features, n_hidden_units), activation()]
        * min(n_hidden_layers, 1),
        *[nn.Linear(n_hidden_units, n_hidden_units), activation()]
        * (n_hidden_layers - 1),
        nn.Linear(n_hidden_units, out_features),
        nn.Identity()
    )
