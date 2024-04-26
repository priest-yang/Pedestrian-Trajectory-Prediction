import torch
import torch.nn as nn

class TraPredModel(nn.Module):
    def __init__(self, input_size=None, lookback=None, num_future_steps=None, layers=[512, 128, 2], hidden_size=512):
        super().__init__()
        self.num_future_steps = num_future_steps
        self.hidden_size = hidden_size

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lookback, batch_first=True, bidirectional=True)
        in_features = hidden_size * 2  # Output from bidirectional LSTM
        # Decoder LSTM

        # The input size for the decoder is the size of the output space
        self.decoder_lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_size, num_layers=num_future_steps, batch_first=True)

        # MLP for processing the output of the decoder
        mlp_layers = []
        layers.insert(0, in_features)  # Insert the input size to the MLP
        for out_features in layers:
            mlp_layers.append(nn.Linear(in_features, out_features))
            mlp_layers.append(nn.LayerNorm(out_features))  # Adding layer normalization
            mlp_layers.append(nn.ReLU())  # Adding ReLU activation function after each Linear layer
            mlp_layers.append(nn.Dropout(0.2))  # Adding dropout
            in_features = out_features
        
        mlp_layers.pop()  # Remove the last ReLU
        mlp_layers.pop()  # Remove the last LayerNorm
        mlp_layers.pop()  # Remove the last Dropout
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, future_steps = None):
        # Encoding
        encoder_out, (h_n, c_n) = self.encoder_lstm(x)

        # Prepare the initial input for the decoder (last observed output)
        decoder_input = encoder_out[:, -1:, :]

        # Decoding
        decoder_out, (h_n, c_n) = self.decoder_lstm(decoder_input, (h_n, c_n))

        # MLP
        outputs = self.mlp(decoder_out)

        # # Decoding
        # outputs = []
        
        # if future_steps is not None:
        #     self.num_future_steps = future_steps
        
        # for _ in range(self.num_future_steps):
        #     decoder_out, (h_n, c_n) = self.decoder_lstm(decoder_input, (h_n, c_n))
        #     decoder_out = self.mlp(decoder_out)
        #     outputs.append(decoder_out)
        #     decoder_input = decoder_out  # Using generated output as next input

        # # Concatenate all outputs
        # outputs = torch.cat(outputs, dim=1)
        return outputs
