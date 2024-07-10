import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseLSTM(nn.Module):
    def __init__(self, input_size=None, lookback=None, layers=[256, 256], hidden_size=64, bidirectional=True, batch_size=None, device='cpu', future_steps=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lookback, batch_first=True, bidirectional=bidirectional, dropout=0.1)
        self.num_layers = lookback
        self.future_steps = future_steps
        self.bi = 2 if bidirectional else 1
        self.batch_size = batch_size
        self.device = device
        
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers * self.bi, batch_size, hidden_size, device=device))  # hidden state
        self.c_0 = nn.Parameter(torch.zeros(self.num_layers * self.bi, batch_size, hidden_size, device=device))  # internal state


        # Adjust MLP to handle sequences
        mlp_layers = []
        neuron_num = hidden_size * self.bi  # *2 removed because we take only the last output of the LSTM
        layers.append(2)  # Predict x, y coordinates at each step
        in_features = neuron_num
        for out_features in layers:
            mlp_layers.append(nn.Linear(in_features, out_features))
            mlp_layers.append(nn.ReLU())  # Adding ReLU activation function after each Linear layer
            in_features = out_features  # Update in_features for the next layer

        # Remove the last ReLU activation since no non-linearity is required at the output
        mlp_layers.pop()

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, future_steps = None):
        if future_steps > self.future_steps:
            raise ValueError(f"Future steps should be less than or equal to {self.future_steps}")
        
        if x.shape[0] != self.batch_size:
            raise ValueError(f"Batch size mismatch. Expected {self.batch_size} but got {x.shape[0]}")
        
        # Encoding
        lstm_out, (h_n, c_n) = self.lstm(x, (self.h_0, self.c_0))
        # Reshape the output to decode trajectories all at once
        # LSTM output: batch_size, seq_len, num_directions * hidden_size
        # We take the output at all sequence steps if needed, or modify accordingly
        last_lstm_out = lstm_out[:, -1, :]  # This gets the last time step; you might want to change it

        # Now predict future steps
        decoder_input = last_lstm_out.unsqueeze(1).repeat(1, self.future_steps, 1)
        decoded = self.mlp(decoder_input)

        # Reshape to (batch_size, future_steps, 2)
        decoded = decoded.view(self.batch_size, self.future_steps, -1)
        if future_steps is not None:
            decoded = decoded[:, :future_steps, :]
        return decoded

# # Model instantiation
# model = TraPredModel(input_size=10, lookback=2, layers=[256, 256], hidden_size=128, bidirectional=True, batch_size=32, device='cpu', future_steps=20)
