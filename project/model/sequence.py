import math
import typing
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from model.metrics import Metric, Seqence, WholeSeqence, Loss


class SequencePredictionModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(input_size, hidden_size)]
            + [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        ).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)
        self._loss = nn.MSELoss(reduction="none")
        self._metrics = [Seqence(-1), Seqence(0)]
        self._iteration_metrics = [Loss(nn.MSELoss()), WholeSeqence()]

    def forward(self, x):
        output = []
        state = [None for _ in self.lstm]
        for i in range(x.shape[1]):
            out, state = self._forward_flow(x[:, i], state)
            output.append(out)
        return torch.stack(output, dim=1), state

    def _forward_flow(self, x, states):
        new_states = []
        for i, (layer, s) in enumerate(zip(self.lstm, states)):
            state = layer(x, s)
            x = state[0] if not i else x + state[0]
            new_states.append(state)
        return self.fc(nn.functional.relu(x)), new_states

    def training_step(self, batch):
        x, y = batch
        y_pred, _ = self(x)
        loss = (self._loss(y_pred, y) / (y + y_pred + 1e-9)).mean()
        self._report(
            "training",
            y_pred.detach().cpu().squeeze(-1).numpy().tolist(),
            y.cpu().squeeze(-1).numpy().tolist(),
            self._iteration_metrics,
        )
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred, _ = self(x)
        self._report(
            "validation",
            y_pred.cpu().squeeze(-1).numpy().tolist(),
            y.cpu().squeeze(-1).numpy().tolist(),
            self._iteration_metrics,
        )

    def test_step(self, batch):
        x, _ = batch
        for sample in x:
            input_seq, target_seq = sample[:50], sample[50:]
            output = self.predict_step(input_seq, len(target_seq))
            self._report(
                "test",
                output,
                target_seq.cpu().squeeze(-1).numpy().tolist(),
                self._metrics,
            )

    def _report(
        self,
        stage: str,
        output: torch.Tensor,
        target: torch.Tensor,
        metrics: typing.List[Metric],
    ):
        if any(x == 0 for x in target):
            return
        for metric in metrics:
            metric_val = metric(output, target)
            self.log(
                f"{stage}/{metric.name}",
                metric_val,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

    def predict_step(self, x, length: int):
        x = x.unsqueeze(0).to(self.device)
        x, hx = self(x)
        x = x[:, -1, :]
        outs = []
        for _ in range(length):
            x, hx = self._forward_flow(x, hx)
            outs.append(x.item())
        return outs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
