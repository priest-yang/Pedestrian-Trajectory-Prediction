import torch
import torch.nn as nn


class TraPredModel(nn.Module):
    def __init__(self, input_size=None, lookback=None, layers=None, hidden_size=50):
        super().__init__()
        if layers is None:
            layers = [128, 128, 64, 2]
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lookback, batch_first=True)

        assert layers[-1] == 2, "The last layer must have 2 output units for the x and y coordinates"
        mlp_layers = []
        in_features = hidden_size  # Output from LSTM becomes input to the MLP
        for out_features in layers:
            mlp_layers.append(nn.Linear(in_features, out_features))
            mlp_layers.append(nn.ReLU())  # Adding ReLU activation function after each Linear layer
            in_features = out_features  # Update in_features for the next layer

        mlp_layers.pop()  # Remove the last ReLU added in the loop

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.mlp(x)
        return x
