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
        self.lstm = nn.LSTMCell(input_size, hidden_size).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)
        self._loss = nn.MSELoss()

    def forward(self, x):
        output = []
        state = None
        for i in range(x.shape[1]):
            out, state = self._forward_flow(x[:, i], state)
            output.append(out)
        return torch.stack(output, dim=1), state

    def _forward_flow(self, x, state):
        state = self.lstm(x, state)
        return self.fc(state[0]), state

    def training_step(self, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        y_pred, _ = self(x)
        loss = self._loss(y_pred, y)
        # mae = torch.abs(y - y_pred).mean()
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_mae", mae)
        return loss

    def predict_step(self, x, length: int):
        t = x
        x = torch.tensor(x.astype("float32")).unsqueeze(0).unsqueeze(-1).to(self.device)
        x, hx = self(x)
        x = x[:, -1, :]
        outs = []
        for _ in range(length):
            x, hx = self._forward_flow(x, hx)
            outs.append(x.item())
        return outs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
