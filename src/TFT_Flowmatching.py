import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.VQVAE import VQVAE

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

class DiffusionDecoder(nn.Module):
    """
    A diffusion–based decoder that iteratively denoises an initial noise sample
    to produce the final output (e.g. action predictions). Conditioning comes from
    an encoded summary (e.g. the last token from the transformer encoder).
    """
    def __init__(self, action_dim, conditioning_dim, num_diffusion_steps=20, num_action_steps=20,
                 num_heads=4, hidden_dim=128):
        """
        Args:
            action_dim (int): Dimensionality of the output (e.g. number of action dimensions).
            conditioning_dim (int): Dimensionality of the conditioning feature.
            num_diffusion_steps (int): How many denoising steps to perform.
            num_action_steps (int): Length of the output sequence.
            num_heads (int): Number of attention heads in the cross-attention.
            hidden_dim (int): Hidden dimension for the denoiser.
        """
        super(DiffusionDecoder, self).__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_action_steps = num_action_steps
        self.hidden_dim = hidden_dim
        
        # Embed a scalar timestep into a hidden vector.
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Project the noisy action sample to the hidden dimension.
        self.query_proj = nn.Linear(action_dim, hidden_dim)
        # Cross–attention: query from current noisy actions, keys/values from conditioning.
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1)
        # Project back to action dimension.
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        # Optional feed-forward refinement.
        self.ffn = nn.Sequential(
            nn.Linear(action_dim, action_dim * 4),
            nn.ReLU(),
            nn.Linear(action_dim * 4, action_dim)
        )
        self.norm = nn.LayerNorm(action_dim)

    def denoise_step(self, conditioning, x_t, timestep):
        """
        One denoising step.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, conditioning_dim).
                        (Ideally, conditioning_dim == hidden_dim.)
            x_t: Tensor of shape (batch, num_action_steps, action_dim) -- current noisy sample.
            timestep: Tensor of shape (batch,) with the current diffusion timestep.
            
        Returns:
            v_t: Denoising estimate (same shape as x_t).
        """
        batch_size = x_t.size(0)
        
        # Embed the scalar timestep.
        t = timestep.view(batch_size, 1)  # (batch, 1)
        t_emb = self.time_embed(t)        # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1)         # (batch, 1, hidden_dim)
        
        # Project the noisy sample and add the time embedding.
        query = self.query_proj(x_t) + t_emb  # (batch, num_action_steps, hidden_dim)
        
        # Prepare for multihead attention: (num_action_steps, batch, hidden_dim).
        query = query.transpose(0, 1)
        key   = conditioning.transpose(0, 1)  # conditioning: (cond_len, batch, hidden_dim)
        value = conditioning.transpose(0, 1)
        
        attn_output, _ = self.cross_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)  # (batch, num_action_steps, hidden_dim)
        
        # Project back to action space.
        v_t = self.output_proj(attn_output)  # Use only the last timestep.
        # Add residual connection and refine.
        v_t = self.norm(x_t + v_t)
        v_t = v_t + self.ffn(v_t)
        return v_t
    
    def predict_v(self, conditioning, x_t, timestep):
        """
        A thin wrapper that predicts v_theta for a given x_t and timestep.
        """
        return self.denoise_step(conditioning, x_t, timestep)

    def forward(self, conditioning, noise=None):
        """
        Run the diffusion process to generate the final output.
        
        Args:
            conditioning: (batch, cond_len, conditioning_dim)
            noise: Optional initial noise of shape (batch, num_action_steps, action_dim).
                   If None, standard Gaussian noise is used.
                   
        Returns:
            x_t: The denoised output (batch, num_action_steps, action_dim)
        """
        batch_size = conditioning.size(0)
        device = conditioning.device
        
        if noise is None:
            noise = torch.randn(batch_size, self.num_action_steps, self.action_dim, device=device)
        
        # Here we use a simple Euler integration schedule:
        dt = -1.0 / self.num_diffusion_steps  # dt < 0 so time goes from 1.0 -> 0.
        time = torch.full((batch_size,), 1.0, device=device, dtype=torch.float32)
        
        x_t = noise
        for _ in range(self.num_diffusion_steps):
            v_t = self.denoise_step(conditioning, x_t, time)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

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
            num_heads=4,  # you can adjust as needed
            hidden_dim=num_hidden
        )

        self.num_steps = num_steps
        self.num_outputs = num_outputs

    def forward(self, x, mask: Optional[torch.Tensor] = None, y_batch=None, influence=False):
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
        conditioning = self.condition_proj(x[:, -1:, :])  # (batch, 1, num_hidden)
        # conditioning = self.condition_proj(x[:, :, :])  # (batch, 1, num_hidden)


        # flow matching during training
        self.device = next(self.parameters()).device
        flow_loss = torch.tensor(0.0, device=self.device)
        # if self.training and y_batch is not None:
        #     # Sample a random time t in [0,1] for flow matching.
        #     t_flow = torch.rand(batch_size, 1, 1, device=self.device)  # shape: (batch, 1, 1)
        #     # Sample noise for the diffusion process.
        #     # (Assuming your diffusion decoder predicts sequences of length equal to future_steps and action dim of 2.)
        #     noise_flow = torch.randn(batch_size, future_steps, self.num_outputs, device=self.device)
        #     # Compute the intermediate sample and target flow using the helper function.
        #     # (Assuming y_batch[:, :future_steps, :2] is your ground truth target.)
        #     x_t_flow, v_target = compute_flow_target(noise_flow, y_batch[:, :future_steps, :self.num_outputs], t_flow)
            
        #     # Predict the velocity (v_pred) using the diffusion decoder.
        #     # Adjust t_flow shape as needed (here squeezed to shape (batch,)).
        #     v_pred = self.diffusion_decoder.predict_v(conditioning, x_t_flow, t_flow.squeeze(-1).squeeze(-1))
            
        #     # Compute the flow matching loss (MSE between predicted flow and target flow).
        #     flow_loss = F.mse_loss(v_pred, v_target)

        #     User_trajectory = self.diffusion_decoder(conditioning)
        #     return User_trajectory, vq_loss, perplexity, flow_loss

        # Run the diffusion process to generate outputs.
        User_trajectory = self.diffusion_decoder(conditioning)
        if influence:
            return User_trajectory
        else:
            return User_trajectory, vq_loss, perplexity, flow_loss


    def influence(self, x):
        User_trajectory = self.forward(x, influence=True)
        return User_trajectory

def compute_flow_target(noise, target, t, schedule_fn=lambda t: t):
    """
    Computes an intermediate sample x_t and its target flow v_target.
    
    Args:
        noise: Tensor of shape (batch, num_action_steps, action_dim)
        target: Ground truth output tensor of shape (batch, num_action_steps, action_dim)
        t: Tensor of shape (batch, 1, 1) with time values in [0,1]
        schedule_fn: A function φ(t) that maps time to interpolation weight.
                     For a linear schedule, schedule_fn(t)=t.
                     
    Returns:
        x_t: The intermediate sample at time t.
        v_target: The target flow, i.e., d x_t / dt.
    """
    # Compute interpolation weight and its derivative.
    phi = schedule_fn(t)              # shape: (batch, 1, 1)
    # For linear schedule, dphi/dt = 1. Otherwise, adjust accordingly.
    dphi_dt = torch.ones_like(phi)    # Modify if using a non-linear schedule.
    
    # Interpolate: x(t) = (1 - φ(t)) * noise + φ(t) * target
    x_t = (1 - phi) * noise + phi * target
    
    # The target flow is: v_target = dφ/dt * (target - noise)
    v_target = dphi_dt * (target - noise)
    
    return x_t, v_target
