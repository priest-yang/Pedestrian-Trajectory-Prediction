import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout_rate=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_input = x
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = norm(x)
        gate = torch.sigmoid(self.gate(x))
        x2 = self.fc2(x)
        return self.fc3(x_input) + gate * x2

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, mask: Optional[torch.Tensor]=None):
        x2 = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + x2
        x2 = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + x2
        return x

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features, num_hidden, num_outputs, num_steps, num_attention_heads=8):
        super(TemporalFusionTransformer, self).__init__()
        self.encoder_grn = GatedResidualNetwork(num_features, num_hidden, num_hidden)
        self.transformer_block = TransformerBlock(num_hidden, num_heads=num_attention_heads, dropout_rate=0.1)
        self.transformer_block2 = TransformerBlock(num_hidden, num_heads=num_attention_heads, dropout_rate=0.1)
        self.decoder_grn = GatedResidualNetwork(num_hidden, num_hidden, num_hidden)
        self.final_linear = nn.Linear(num_hidden, num_outputs * num_steps)
        self.num_steps = num_steps
        self.num_outputs = num_outputs

    def forward(self, x,  mask: Optional[torch.Tensor]=None):
        if len(x.shape) == 2:
            # bacth_size = 1, in real-time mode
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.shape
        x = self.encoder_grn(x)
        x = x.permute(1, 0, 2)  # Prepare shape for nn.MultiheadAttention (seq_len, batch_size, num_features)
        x = self.transformer_block(x, mask=mask)
        x = self.transformer_block2(x, mask=mask)
        x = x.permute(1, 0, 2)  # Revert shape to (batch_size, seq_len, num_features)
        x = self.decoder_grn(x)
        x = self.final_linear(x)
        x = x.view(batch_size, seq_len, self.num_outputs, self.num_steps) # Reshape to (batch_size, seq_len, 2 [UserX, UserY], 20 [time_steps])
        return x[:, -1, :, :].permute(0, 2, 1)  # [batch_size, num_steps, num_outputs]
