import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class SequencePredictionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, 1:]
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        # mae = torch.abs(y - y_pred).mean()
        self.log("train_loss", loss)
        # self.log("train_mae", mae)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
