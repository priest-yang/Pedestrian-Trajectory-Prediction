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

class DiffusionDecoder(nn.Module):
    def __init__(self, action_dim, conditioning_dim, num_diffusion_steps=10,
                 num_action_steps=20, num_heads=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.hidden_dim = hidden_dim
        self.num_action_steps = num_action_steps

        # Improved beta schedule
        betas = cosine_beta_schedule(num_diffusion_steps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))

        # Enhanced time embedding
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
        # Time embedding
        t_emb = self.time_embed(t)  # [batch, h_dim]
        
        # Combine input with time embeddings
        x_proj = self.input_proj(torch.cat([x_t, t_emb.unsqueeze(1).expand(-1, x_t.size(1), -1)], -1))
        
        # Process through transformer blocks
        h = x_proj
        for block in self.blocks:
            # Cross-attention
            attn_out, _ = block['cross_attn'](
                h, conditioning, conditioning,
            )
            h = h + attn_out
            h = block['norm1'](h)
            
            # MLP
            h = h + block['mlp'](h)
            h = block['norm2'](h)
        
        return self.out(h)
    
    def influence(self, conditioning, device):
        """
        Runs the reverse diffusion process and returns a list of intermediate denoised trajectories.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, conditioning_dim)
            device: torch.device.
            
        Returns:
            intermediates: A list of tensors, each of shape (batch, num_action_steps, action_dim),
                           representing the denoised trajectory at each diffusion step.
        """
        batch_size = conditioning.size(0)
        x = torch.randn(batch_size, self.num_action_steps, self.action_dim, device=device)
        intermediates = []
        # Reverse diffusion: record intermediate results at each step.
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            epsilon_pred = self.forward(conditioning, x, t_tensor)
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alpha_bars[t]
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred) \
                + torch.sqrt(beta_t) * noise
            # Save a clone of the current state.
            intermediates.append(x.clone())
        return intermediates
    

    def sample(self, conditioning, device):
        """
        Generate a trajectory by running the reverse diffusion process.
        
        Args:
            conditioning: Tensor of shape (batch, cond_len, conditioning_dim)
            device: torch.device to run the sampling on.
            
        Returns:
            x: Generated trajectory of shape (batch, num_action_steps, action_dim)
        """
        batch_size = conditioning.size(0)
        # Start from standard Gaussian noise.
        x = torch.randn(batch_size, self.num_action_steps, self.action_dim, device=device)
        # Reverse diffusion process (using the DDPM update)
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            epsilon_pred = self.forward(conditioning, x, t_tensor)
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alpha_bars[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0.0
            # DDPM reverse update:
            x = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred) \
                + torch.sqrt(beta_t) * noise
        return x
    
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
        conditioning = self.condition_proj(x[:, -1:, :])  # (batch, 1, num_hidden)
        # conditioning = self.condition_proj(x[:, :, :])  # (batch, 1, num_hidden)



        # flow matching during training
        self.device = next(self.parameters()).device
        flow_loss = torch.tensor(0.0, device=self.device)

        
        if influence:
            if return_all:
                return self.diffusion_decoder.influence(conditioning, self.device)
            return self.diffusion_decoder.influence(conditioning, self.device)[-1]
        else:
            if self.training:
                diff_loss = self.decoder_train_step(conditioning, y_batch, self.device)
                return diff_loss, vq_loss, perplexity, flow_loss
            
            return User_trajectory, vq_loss, perplexity, flow_loss


    def influence(self, x):
        User_trajectory = self.forward(x, influence=True)
        return User_trajectory
    
    def decoder_train_step(self, conditioning, y_batch, device):
        """
        Performs one training step for the diffusion self.diffusion_decoder.
        
        Args:
            self.diffusion_decoder: Instance of DiffusionDecoder.
            conditioning: Conditioning tensor (batch, cond_len, conditioning_dim)
                        (e.g., output from an encoder that has been projected to hidden_dim).
            y_batch: Ground truth trajectory (batch, num_action_steps, action_dim).
            device: torch.device.
            
        Returns:
            loss: The MSE loss between predicted and true noise.
        """
        batch_size = y_batch.size(0)
        # Sample random timesteps for each example in the batch.
        t = torch.randint(0, self.diffusion_decoder.num_diffusion_steps, (batch_size,), device=device)
        # Gather the corresponding alpha_bar for each timestep.
        alpha_bar_t = self.diffusion_decoder.alpha_bars[t].view(batch_size, 1, 1)  # (batch, 1, 1)
        # Sample noise to add.
        noise = torch.randn_like(y_batch)
        # Create the noisy trajectory: x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
        x_t = torch.sqrt(alpha_bar_t) * y_batch + torch.sqrt(1.0 - alpha_bar_t) * noise
        # The network predicts the noise given x_t and the conditioning.
        noise_pred = self.diffusion_decoder(conditioning, x_t, t)
        # MSE loss between the predicted noise and the actual noise.
        loss = F.mse_loss(noise_pred, noise)
        return loss

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



class DecayLoss(nn.Module):
    def __init__(self, num_steps, baseline_loss_fn=nn.L1Loss()):
        super(DecayLoss, self).__init__()
        # Weight decreases as we move further into the future
        self.weights = torch.linspace(1.0, 1.0, num_steps)
        self.baseline_loss_fn = baseline_loss_fn
        

    def forward(self, predictions, targets):
        loss = 0
        for i in range(predictions.shape[1]):
            loss += self.weights[i] * self.baseline_loss_fn(predictions[:, i], targets[:, i])
        return loss
    
    
baseline_loss_fn = nn.L1Loss() #nn.MSELoss()
loss_fn = DecayLoss(future_steps, baseline_loss_fn=baseline_loss_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

vqvae = VQVAE(input_dim=feature_dim, hidden_dim=512, num_embeddings=128, embedding_dim=128, commitment_cost=0.25)

model = TemporalFusionTransformerDiffusion(num_features=feature_dim, num_hidden=128, num_outputs=2, num_steps=future_steps, diffusion_steps=10, vqvae=vqvae)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
model.to(device)


# Parameters
n_epochs = 50
eval_step = 2000
save_every = 10000
patience = 8  # Number of evaluations to wait for improvement
cooldown = 4  # Evaluations to wait after an improvement before counting non-improvements
smooth_factor = 0.6  # Smoothing factor for moving average
lambda_flow = 1e-3  # Weight for flow matching loss

# Setup
train_all = len(train)
model_name = "TFT_Flowmatching"
from collections import defaultdict
loss_all = defaultdict(list)
best_test_rmse = float('inf')
early_stopping_counter = 0
cooldown_counter = cooldown

now = datetime.now()
folder_name = now.strftime("%b%d_%H-%M-%S")
print(f"Saving model at ../model/{model_name}/{folder_name}")

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(len(train) * n_epochs, 50000), eta_min=1e-6)

# Initialize moving average
moving_avg_test_rmse = None

# Training loop
for epoch in range(n_epochs):
    model.train()
    for step, (X_batch, y_batch) in tqdm(enumerate(train), total=train_all):
        X_batch = X_batch.float().to(device)
        y_batch = y_batch.float().to(device)
        
        current_pos_input = X_batch[:, -1, :2].clone().unsqueeze(1).repeat(1, lookback, 1)
        current_pos_output = X_batch[:, -1, :2].clone().unsqueeze(1).repeat(1, future_steps, 1)
        X_batch[:, :, :2] = X_batch[:, :, :2] - current_pos_input
        y_batch[:, :, :2] = y_batch[:, :, :2] - current_pos_output


        # residual
        # X_batch[:, 1:, :2] = X_batch[:, 1:, :2] - X_batch[:, :-1, :2].clone()
        # X_batch[:, 0, :2] = 0
        # y_batch[:, 1:, :2] = y_batch[:, 1:, :2] - y_batch[:, :-1, :2].clone()
        # y_batch[:, 0, :2] = 0

        optimizer.zero_grad()
        
        # y_pred, vq_loss, perplexity, flow_loss = model(X_batch, y_batch=y_batch)
        # loss = loss_fn(y_pred[:, :future_steps, :2], y_batch[:, :future_steps, :2])
        diff_loss, vq_loss, perplexity, flow_loss = model(X_batch, y_batch[:, :future_steps, :2])


        loss_all['diff_loss'].append(diff_loss.item())
        loss_all['vq_loss'].append(vq_loss.item() * 10)
        # add vq_loss
        loss = 10 * vq_loss + diff_loss
        # add flow_loss
        loss += lambda_flow * flow_loss
        loss_all['flow_loss'].append(flow_loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Save model
        if (epoch * train_all + step + 1) % save_every == 0:
            os.makedirs(f'../model/{model_name}/{folder_name}', exist_ok=True)
            save_path = f"../model/{model_name}/{folder_name}/model_{epoch * train_all + step + 1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

        # Validation and early stopping
        if (epoch * train_all + step + 1) % eval_step == 0:
            model.eval()
            test_rmse_all = []
            with torch.no_grad():
                for X_test_batch, y_test_batch in test:
                    X_test_batch = X_test_batch.float().to(device)
                    y_test_batch = y_test_batch.float().to(device)
                    
                    current_pos_input = X_test_batch[:, -1, :2].clone().unsqueeze(1).repeat(1, lookback, 1)
                    current_pos_output = X_test_batch[:, -1, :2].clone().unsqueeze(1).repeat(1, future_steps, 1)
                    X_test_batch[:, :, :2] = X_test_batch[:, :, :2] - current_pos_input
                    y_test_batch[:, :, :2] = y_test_batch[:, :, :2] - current_pos_output
                    
                    
                    # # residual
                    # X_test_batch[:, 1:, :2] = X_test_batch[:, 1:, :2] - X_test_batch[:, :-1, :2].clone()
                    # X_test_batch[:, 0, :2] = 0
                    # y_test_batch[:, 1:, :2] = y_test_batch[:, 1:, :2] - y_test_batch[:, :-1, :2].clone()
                    # y_test_batch[:, 0, :2] = 0
                    
                    y_pred_test = model(X_test_batch, influence=True)
                    loss_test = loss_fn(y_pred_test[:, :future_steps, :2], y_test_batch[:, :future_steps, :2])
                    test_rmse = torch.sqrt(loss_test)
                    if not torch.isnan(test_rmse):
                        test_rmse_all.append(test_rmse.item())
            
            current_rmse = sum(test_rmse_all) / len(test_rmse_all)
            if moving_avg_test_rmse is None:
                moving_avg_test_rmse = current_rmse
            else:
                moving_avg_test_rmse = smooth_factor * current_rmse + (1 - smooth_factor) * moving_avg_test_rmse

            print(f"Steps {epoch * train_all + step + 1}: test RMSE {current_rmse:.4f}, moving average RMSE {moving_avg_test_rmse:.4f}")

            # Check if the moving average RMSE is better; if not, increment counter
            if moving_avg_test_rmse < best_test_rmse:
                best_test_rmse = moving_avg_test_rmse
                early_stopping_counter = 0  # Reset counter
                cooldown_counter = cooldown  # Reset cooldown
                # Optionally save the best model
                os.makedirs(f'../model/{model_name}/{folder_name}', exist_ok=True)
                best_model_path = f"../model/{model_name}/{folder_name}/best_model.pt"
                torch.save(model.state_dict(), best_model_path)
            else:
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                else:
                    early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print(f"Stopping early at epoch {epoch+1}, step {step+1}")
                break

            model.train()
        
    if early_stopping_counter >= patience:
        break

print("Training complete.")