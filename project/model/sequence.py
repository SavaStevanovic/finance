import collections
import math
import typing
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.metrics import Metric, Seqence, WholeSeqence, Loss
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("figure", figsize=(50, 25))


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
        self._mean_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size), nn.Softplus()
        ).to(self.device)
        self._std_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size), nn.Softplus()
        ).to(self.device)
        self._metrics = [Seqence(-1), Seqence(0)]
        self._iteration_metrics = [Loss(nn.MSELoss()), WholeSeqence()]

    def forward(self, x):
        output = []
        state = [None for _ in self.lstm]
        for i in range(x.shape[1]):
            out, state = self._forward_flow(x[:, i], state)
            output.append(out)
        return torch.stack([x.mean for x in output], 1), state, output

    def _forward_flow(self, x, states):
        new_states = []
        for i, (layer, s) in enumerate(zip(self.lstm, states)):
            state = layer(x, s)
            x = state[0] if not i else x + state[0]
            new_states.append(state)
        mean = self._mean_layer(x)
        std = self._std_layer(x)
        pred = torch.distributions.Normal(mean, std)
        return pred, new_states

    def training_step(self, batch):
        x, y = batch
        y_pred, _, dist = self(x)
        loss = torch.stack(
            [-dist[i].log_prob(y[:, i, :]) for i in range(len(dist))]
        ).mean()

        self._report(
            "training",
            y_pred.detach().cpu().squeeze(-1).numpy(),
            y.cpu().squeeze(-1).numpy(),
            self._iteration_metrics,
        )
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred, _, _ = self(x)
        self._report(
            "validation",
            y_pred.cpu().squeeze(-1).numpy(),
            y.cpu().squeeze(-1).numpy(),
            self._iteration_metrics,
        )

    def test_step(self, batch):
        x, _ = batch
        for sample in x:
            input_seq, target_seq = sample[:50], sample[50:]
            output, _ = self._predict_step(input_seq, len(target_seq))
            self._report(
                "test",
                output,
                target_seq.cpu().squeeze(-1).numpy(),
                self._metrics,
            )

    def backtest(self, sample):
        sample_len = 500
        offset = sample[:-sample_len][:, 0].sum().item()
        sample = sample[-sample_len:]
        length = int(len(sample) // 2)
        input_seq, target_seq = sample[:length], sample[length:]
        output, sdvs = self._predict_step(input_seq, len(target_seq))
        sdvs = sdvs[:, 0]
        input_seq = input_seq[:, 0].numpy().cumsum() + offset
        target_seq = target_seq[:, 0].numpy().cumsum()
        output = output[:, 0].cumsum()
        output += input_seq[-1]
        target_seq += input_seq[-1]
        plt.xlabel("Time")
        plt.ylabel("Value")
        print(plt.rcParamsDefault["figure.figsize"])
        plt.plot(input_seq, label=f"input")
        pred_ids = np.arange(len(input_seq), len(input_seq) + len(target_seq))
        plt.plot(pred_ids, target_seq, label=f"target", linewidth=2)
        plt.plot(pred_ids, output, label=f"output", linewidth=2.0)
        plt.plot(pred_ids, output + sdvs, label=f"output + sdvs", linewidth=1)
        plt.plot(pred_ids, output - sdvs, label=f"output - sdvs", linewidth=1)
        plt.legend()

        plt.savefig("msft.png")

    def _report(
        self,
        stage: str,
        output: torch.Tensor,
        target: torch.Tensor,
        metrics: typing.List[Metric],
    ):
        if not target.all():
            return
        for metric in metrics:
            metric_val = metric(output, target)
            if metric_val.shape:
                log_data = {
                    f"{stage}/{metric.name}/{i}": val
                    for i, val in enumerate(metric_val)
                }
                self.log_dict(
                    log_data,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
            else:
                self.log(
                    f"{stage}/{metric.name}",
                    metric_val,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

    def _predict_step(self, x, length: int):
        x = x.unsqueeze(0).to(self.device)
        x, hx, _ = self(x)
        x = x[:, -1, :]
        outs = []
        sdvs = []
        for _ in range(length):
            x, hx = self._forward_flow(x, hx)
            outs.append(x.mean[0].cpu().detach().numpy())
            sdvs.append(x.stddev[0].cpu().detach().numpy())
            x = x.mean
        return np.stack(outs, 0), np.stack(sdvs, 0)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)
