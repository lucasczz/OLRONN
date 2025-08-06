from torch import nn
import torch


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
        nn.Identity(),
    )


def get_autoencoder(
    in_features, n_hidden_units, dropout=0.1, n_hidden_layers=1, activation=nn.ReLU
):
    return nn.Sequential(
        *[nn.Linear(in_features, in_features), activation()] * min(n_hidden_layers, 1),
        *[nn.Linear(in_features, in_features), activation()] * (n_hidden_layers - 1),
        nn.Linear(in_features, n_hidden_units),
        activation(),
        nn.Linear(n_hidden_units, in_features),
        activation(),
        *[nn.Linear(in_features, in_features), activation()] * (n_hidden_layers - 1),
        *[nn.Linear(in_features, in_features), activation()] * min(n_hidden_layers, 1),
    )


def get_conditional_autoencoder(
    in_features, n_classes, n_hidden_units=128, n_hidden_layers=1
):
    class ConditionalAutoencoder(nn.Module):
        def __init__(self, in_features, n_classes, n_hidden_units, n_hidden_layers):
            super().__init__()
            self.embedding = nn.Embedding(n_classes, n_hidden_units // 8)
            self.encoder = nn.Sequential(
                nn.Linear(in_features, n_hidden_units),
                nn.ReLU(),
                *[
                    nn.Sequential(nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU())
                    for _ in range(n_hidden_layers - 1)
                ],
                nn.Linear(n_hidden_units, n_hidden_units),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(n_hidden_units + n_hidden_units // 8, n_hidden_units),
                nn.ReLU(),
                *[
                    nn.Sequential(nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU())
                    for _ in range(n_hidden_layers - 1)
                ],
                nn.Linear(n_hidden_units, in_features),
            )
            self.cond_features = n_hidden_units // 8

        def forward(self, x, y):
            # y is a label, encode using embedding
            y_emb = self.embedding(y)
            z = self.encoder(x)
            z_cat = torch.cat([z, y_emb], dim=-1)
            x_recon = self.decoder(z_cat)
            return x_recon

    return ConditionalAutoencoder(
        in_features, n_classes, n_hidden_units, n_hidden_layers
    )
