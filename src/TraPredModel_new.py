import torch
import torch.nn as nn
from torch.autograd import Variable

# LSTM in encoder-decoder

class TraPredModel(nn.Module):
    def __init__(self, input_size=None, lookback=None, layers=[256, 256], hidden_size=64, bidirectional=True, batch_size=None, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = lookback
        self.batch_size = batch_size
        self.device = device

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lookback, batch_first=True, bidirectional=bidirectional, dropout=0.1)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lookback, batch_first=True, bidirectional=bidirectional, dropout=0.1)

        self.bi = 2 if bidirectional else 1
        neuron_num = hidden_size * self.bi
        
        if batch_size is None:
            raise ValueError("Please provide the batch size")
        else:
            self.batch_size = batch_size
        
        self.h_0 = Variable(torch.zeros(self.num_layers * self.bi, batch_size, hidden_size, device=device))  # hidden state
        self.c_0 = Variable(torch.zeros(self.num_layers * self.bi, batch_size, hidden_size, device=device))  # internal state

        mlp_layers = []
        in_features = neuron_num
        layers.append(input_size)
        for out_features in layers:
            mlp_layers.append(nn.Linear(in_features, out_features))
            mlp_layers.append(nn.ReLU())  # Adding ReLU activation function after each Linear layer
            mlp_layers.append(nn.Dropout(0.2))  # Adding dropout layer
            in_features = out_features  # Update in_features for the next layer
        
        mlp_layers.pop()  # Remove the last ReLU added in the loop
        mlp_layers.pop()  # Remove the last Dropout added in the loop
        
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, future_steps=5):
        if x.shape[0] != self.batch_size:
            raise ValueError(f"Batch size mismatch. Expected {self.batch_size} but got {x.shape[0]}")
        
        # Encoding
        encoder_out, (h_n, c_n) = self.encoder_lstm(x, (self.h_0, self.c_0))

        # Prepare the decoder input (initially zero)
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)).to(self.device)
        outputs = []

        # Decoding for the required number of future steps
        for _ in range(future_steps):
            decoder_out, (h_n, c_n) = self.decoder_lstm(decoder_input, (h_n, c_n))
            output = self.mlp(decoder_out[:, -1, :])
            outputs.append(output.unsqueeze(1))
            decoder_input = output.unsqueeze(1)  # Feeding the output as the next input

        return torch.cat(outputs, dim=1)
    
    
    


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using {device}")

# model = TraPredModel(input_size=feature_dim, lookback=lookback, \
#     hidden_size=128, bidirectional=True, batch_size=batch_size, device=device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-6)
# loss_fn = nn.MSELoss()


# model.to(device)
# n_epochs = 10
# eval_step = 500
# future_steps = 20

# # model = TraPredModel(input_size=numeric_df.shape[1], lookback=lookback)

# save_every = 10000
# train_all = len(train)

# loss_all = []

# now = datetime.now()
# folder_name = now.strftime("%b%d_%H-%M-%S")
# os.makedirs(f'../model/{folder_name}', exist_ok=True)


# for epoch in range(n_epochs):
#     model.train()
#     for step, (X_batch, y_batch) in tqdm(enumerate(train), total = train_all):
#         X_batch = X_batch.float().to(device)
#         y_batch = y_batch.float().to(device)
#         optimizer.zero_grad()
        
#         if X_batch.shape[0] != model.batch_size:
#             continue
#         y_pred = model(X_batch, future_steps=future_steps)
#         loss = torch.mean(loss_fn(y_pred[:, :future_steps, :2], y_batch[:, :future_steps, :2].squeeze(1)))
        
#         if torch.isnan(loss):
#             print("Loss is NaN")
#             continue
#         loss_all.append(loss.item())
#         loss.backward()
#         # Apply gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         # Validation
#         if (epoch * train_all + step + 1) % save_every == 0:
#             print(f"Saving model at epoch {epoch+1}, step {step+1}")

#             torch.save(model.state_dict(), f"../model/{folder_name}/model_{epoch+1}_{step+1}.pt")
        
#         if (epoch * train_all + step + 1) % eval_step == 0:
#             print(f"Start testing")
#             with torch.no_grad():
#                 model.eval()
#                 all_test = len(test)
#                 test_rmse_all = []
#                 for X_test_batch, y_test_batch in tqdm(test):
#                     if X_test_batch.shape[0] != model.batch_size:
#                         continue
#                     X_test_batch = X_test_batch.float().to(device)
#                     y_test_batch = y_test_batch.float().to(device)
#                     y_pred = model(X_test_batch, future_steps=future_steps)
#                     test_rmse = torch.mean(loss_fn(y_pred[:, :future_steps, :2], y_test_batch[:, :future_steps, :2]))
#                     test_rmse = torch.sqrt(test_rmse)
#                     if not torch.isnan(test_rmse):
#                         test_rmse_all.append(test_rmse.item())

#                 print("Epoch %d: test RMSE %.4f" % (epoch+1, sum(test_rmse_all)/all_test))
            
#             model.train()
        

