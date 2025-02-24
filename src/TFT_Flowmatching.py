import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.VQVAE import VQVAE
import math

###############################################
# Original Blocks (with minor efficiency tweaks)
###############################################

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout_rate=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
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
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
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

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (seq_len, batch, hidden_size)
        x2 = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + x2
        x2 = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + x2
        return x

###############################################
# Diffusion–based Decoder
###############################################

class SinusoidalTimeEmbedding(nn.Module):
    """
    Computes a sinusoidal embedding for a scalar timestep.
    """
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        # t: Tensor of shape (batch,) or (batch, 1)
        if len(t.shape) == 1:
            t = t.unsqueeze(1)  # (batch, 1)
        half_dim = self.embedding_dim // 2
        # Compute constant
        emb_factor = math.log(10000) / (half_dim - 1)
        # Create a tensor of shape (half_dim,)
        dims = torch.arange(half_dim, device=t.device, dtype=t.dtype)
        # (batch, half_dim)
        emb = t * torch.exp(-dims * emb_factor)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # If embedding_dim is odd, pad an extra zero.
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (batch, embedding_dim)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)



def compute_flow_target(noise, y_batch, t):
    """
    Compute the intermediate sample x_t and target velocity for flow-matching.
    
    Args:
        noise: Tensor of shape (batch, num_action_steps, action_dim), noise sample
        y_batch: Tensor of shape (batch, num_action_steps, action_dim), ground truth actions
        t: Tensor of shape (batch, 1, 1), time steps
    
    Returns:
        x_t: Intermediate sample at time t
        v_target: Target velocity
    """
    t = t.view(-1, 1, 1)  # Ensure t is [batch, 1, 1]
    x_t = t * noise + (1 - t) * y_batch
    v_target = noise - y_batch
    return x_t, v_target

class DiffusionDecoder(nn.Module):
    def __init__(self, action_dim, conditioning_dim, num_diffusion_steps=10,
                 num_action_steps=20, num_heads=4, hidden_dim=128, num_layers=2, noise_weight=1):
        super().__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps  # Number of integration steps
        self.hidden_dim = hidden_dim
        self.num_action_steps = num_action_steps
        self.noise_weight = noise_weight

        # self.initial_guess_proj = nn.Linear(hidden_dim, action_dim * num_action_steps)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        
        # Input projection with time conditioning
        self.input_proj = nn.Linear(action_dim + hidden_dim, hidden_dim)

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            }) for _ in range(num_layers)
        ])

        # Final output projection
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, conditioning, x_t, t):
        """
        Predicts the velocity given conditioning, intermediate sample x_t, and time t.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, hidden_dim)
            x_t: Tensor of shape (batch, num_action_steps, action_dim)
            t: Tensor of shape (batch,) with time values in [0,1], dtype float
        
        Returns:
            v_pred: Predicted velocity of shape (batch, num_action_steps, action_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [batch, h_dim]
        
        # Combine input with time embeddings
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, x_t.size(1), -1)
        x_with_time = torch.cat([x_t, t_emb_expanded], dim=-1)  # [batch, num_action_steps, action_dim + h_dim]
        x_proj = self.input_proj(x_with_time)  # [batch, num_action_steps, h_dim]
        
        # Process through transformer blocks
        h = x_proj
        for block in self.blocks:
            h_norm = block['norm1'](h)
            attn_out, _ = block['cross_attn'](h_norm, conditioning, conditioning)
            h = h + attn_out
            
            h = block['norm2'](h)
            h = h + block['mlp'](h)
        
        return self.out(h)  # [batch, num_action_steps, action_dim]
    
    def decoder_train_step(self, conditioning, y_batch, device):
        """
        Performs one training step for the flow-matching decoder.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, conditioning_dim)
            y_batch: Ground truth trajectory (batch, num_action_steps, action_dim)
            device: torch.device
        
        Returns:
            loss: The MSE loss between predicted and target velocity
        """
        batch_size = y_batch.size(0)
        # Sample t uniformly from [0,1]
        t = torch.rand(batch_size, device=device)  # [batch]
        t = t.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
        
        # Sample noise
        noise = torch.randn_like(y_batch) * self.noise_weight
        
        # Compute x_t and v_target
        x_t, v_target = compute_flow_target(noise, y_batch, t)
        
        # Predict velocity
        v_pred = self.forward(conditioning, x_t, t.squeeze(2).squeeze(1))  # t: [batch]
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        return loss
    
    def sample(self, conditioning, device):
        """
        Generate a trajectory by integrating the velocity field backwards from t=1 to t=0.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, conditioning_dim)
            device: torch.device
        
        Returns:
            x: Generated trajectory of shape (batch, num_action_steps, action_dim)
        """
        batch_size = conditioning.size(0)
        # initial_guess_flat = self.initial_guess_proj(conditioning.mean(dim=1))  # Pool over seq_len
        # initial_guess = initial_guess_flat.view(batch_size, self.num_action_steps, self.action_dim)
        # x = initial_guess + torch.randn_like(initial_guess) * self.noise_weight
        x = torch.randn(batch_size, self.num_action_steps, self.action_dim, device=device) * self.noise_weight
        dt = -1.0 / self.num_diffusion_steps  # Negative dt for backward integration
        for i in range(self.num_diffusion_steps):
            t = 1.0 + i * dt  # t decreases from 1.0 to almost 0
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float)
            v_pred = self.forward(conditioning, x, t_tensor)
            x = x + v_pred * dt  # Since dt < 0, moves x towards data
        return x
    
    def influence(self, conditioning, device):
        """
        Runs the flow-matching integration process and returns a list of intermediate trajectories.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, conditioning_dim)
            device: torch.device
        
        Returns:
            intermediates: A list of tensors, each of shape (batch, num_action_steps, action_dim),
                           representing the trajectory at each integration step
        """
        batch_size = conditioning.size(0)
        x = torch.randn(batch_size, self.num_action_steps, self.action_dim, device=device) * self.noise_weight
        intermediates = []
        dt = -1.0 / self.num_diffusion_steps  # Negative dt for backward integration
        for i in range(self.num_diffusion_steps):
            t = 1.0 + i * dt  # t decreases from 1.0 to almost 0
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float)
            v_pred = self.forward(conditioning, x, t_tensor)
            x = x + v_pred * dt  # Since dt < 0, moves x towards data
            intermediates.append(x.clone())
        return intermediates
    
###############################################
# Modified Temporal Fusion Transformer with Diffusion Decoder
###############################################

class TemporalFusionTransformerDiffusion(nn.Module):
    def __init__(self, num_features, num_hidden, num_outputs, num_steps,
                 num_attention_heads=8, diffusion_steps=10, vqvae: VQVAE = None):
        """
        Args:
            num_features (int): Number of input features.
            num_hidden (int): Hidden dimension size.
            num_outputs (int): Dimensionality of each output (e.g. action dimension).
            num_steps (int): Desired output sequence length (e.g. number of action steps).
            num_attention_heads (int): Number of heads for the transformer blocks.
            diffusion_steps (int): Number of diffusion (denoising) steps.
        """
        super(TemporalFusionTransformerDiffusion, self).__init__()
        if vqvae is None:
            self.vqvae = VQVAE(input_dim=feature_dim, hidden_dim=512, num_embeddings=128, embedding_dim=128, commitment_cost=0.25)
        else:
            self.vqvae = vqvae
        num_features = num_features + self.vqvae.encoder.fc2.out_features
        self.encoder_grn = GatedResidualNetwork(num_features, num_hidden, num_hidden)
        self.transformer_block = TransformerBlock(num_hidden, num_heads=num_attention_heads, dropout_rate=0.1)
        self.transformer_block2 = TransformerBlock(num_hidden, num_heads=num_attention_heads, dropout_rate=0.1)
        
        # To condition the diffusion process we project the transformer output.
        self.condition_proj = nn.Linear(num_hidden, num_hidden)
        # Diffusion decoder: we set action_dim=num_outputs and produce a sequence of length num_steps.
        self.diffusion_decoder = DiffusionDecoder(
            action_dim=num_outputs,
            conditioning_dim=num_hidden,
            num_diffusion_steps=diffusion_steps,
            num_action_steps=num_steps,
            num_heads=num_attention_heads,  
            hidden_dim=num_hidden, 
            num_layers=2,  # you can adjust as needed
            noise_weight=0.5  # you can adjust as needed
        )

        self.num_steps = num_steps
        self.num_outputs = num_outputs

    def forward(self, x, y_batch=None , mask: Optional[torch.Tensor] = None, influence=False, return_all=False):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, num_features).
            mask: Optional attention mask for the transformer blocks.
            
        Returns:
            actions: Tensor of shape (batch, num_steps, num_outputs)
        """
        # If given a 2D input, add a batch dimension.
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, _ = x.shape

        # VQ-VAE
        x_recon, vq_loss, perplexity, embedding = self.vqvae(x)
        x = torch.cat((x, embedding), dim=-1)
        
        # Encoder GRN.
        x = self.encoder_grn(x)  # (batch, seq_len, num_hidden)
        
        # Transformer expects (seq_len, batch, hidden_size).
        x = x.permute(1, 0, 2)
        x = self.transformer_block(x, mask=mask)
        x = self.transformer_block2(x, mask=mask)
        x = x.permute(1, 0, 2)  # back to (batch, seq_len, num_hidden)
        
        # Use a summary of the encoder output as conditioning.
        # Here we use the last time–step (you might also try an average or more complex pooling).

        # attention
        # attention_weights = torch.softmax(torch.mean(x, dim=-1), dim=1).unsqueeze(-1)
        # pooled_output = torch.sum(attention_weights * x, dim=1, keepdim=True)

        # conditioning = self.condition_proj(pooled_output)  # (batch, 1, num_hidden)
        conditioning = self.condition_proj(x[:, -1:, :])  # (batch, 1, num_hidden)
        # conditioning = self.condition_proj(x[:, :, :])  # (batch, 1, num_hidden)



        # flow matching during training
        self.device = next(self.parameters()).device
        
        if influence:
            if return_all:
                return self.diffusion_decoder.influence(conditioning, self.device)
            return self.diffusion_decoder.influence(conditioning, self.device)[-1]
        else:
            if self.training:
                diff_loss = self.diffusion_decoder.decoder_train_step(conditioning, y_batch, self.device)
                return diff_loss, vq_loss, perplexity
            

    def influence(self, x):
        User_trajectory = self.forward(x, influence=True)
        return User_trajectory

