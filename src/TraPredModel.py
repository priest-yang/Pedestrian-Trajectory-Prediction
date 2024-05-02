import torch
import torch.nn as nn


class TraPredModel(nn.Module):
    def __init__(self, input_size = None, lookback = None,  layers=[512, 256, 2], hidden_size = 64, bidirectional = True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lookback, batch_first=True, bidirectional=bidirectional, dropout=0.1)

        assert layers[-1] == 2, "The last layer must have 2 output units for the x and y coordinates"
        bi = 2 if bidirectional else 1
        neuron_num = hidden_size * 2 + hidden_size * bi

        mlp_layers = []
        in_features = neuron_num
        for out_features in layers:
            mlp_layers.append(nn.Linear(in_features, out_features))
            mlp_layers.append(nn.LayerNorm(out_features))  # Adding layer normalization
            mlp_layers.append(nn.ReLU())  # Adding ReLU activation function after each Linear layer
            mlp_layers.append(nn.Dropout(0.2))  # Adding dropout layer
            in_features = out_features  # Update in_features for the next layer
        
        mlp_layers.pop() # Remove the last ReLU added in the loop
        mlp_layers.pop() # Remove the last LayerNorm added in the loop
        mlp_layers.pop() # Remove the last Dropout added in the loop
        
        self.mlp = nn.Sequential(*mlp_layers)




    def forward(self, x): 
        lstm_out, (h_n, c_n) = self.lstm(x)
        batch_first_hidden = torch.cat((c_n.permute(1, 0, 2), h_n.permute(1, 0, 2)), dim=2)

        last_batch_hidden = batch_first_hidden[:, -1, :]
        last_out = lstm_out[:, -1, :]
        latent_all = torch.cat((last_batch_hidden, last_out), dim=1)

        out = self.mlp(latent_all)
        return out