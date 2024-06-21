import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        gate = torch.sigmoid(self.gate(x1))
        x2 = self.fc2(x1)
        return x + gate * x2

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features, num_hidden, num_outputs):
        super(TemporalFusionTransformer, self).__init__()
        self.encoder_grn = GatedResidualNetwork(num_features, num_hidden, num_hidden)
        self.self_attention = nn.MultiheadAttention(num_hidden, num_heads=8)
        self.decoder_grn = GatedResidualNetwork(num_hidden, num_hidden, num_hidden)
        self.final_linear = nn.Linear(num_hidden, num_outputs)

    def forward(self, x, mask=None):
        x = self.encoder_grn(x)
        x, _ = self.self_attention(x, x, x, key_padding_mask=mask)
        x = self.decoder_grn(x)
        return self.final_linear(x)