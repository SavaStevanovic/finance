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

matplotlib.rc("figure", figsize=(200, 50))
from sklearn.preprocessing import RobustScaler


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
        self._mean_layer = nn.Sequential(nn.Linear(hidden_size, output_size)).to(
            self.device
        )
        self._metrics = [Seqence(-1), Seqence(0)]
        self._iteration_metrics = [Loss(nn.L1Loss()), WholeSeqence()]

    def forward(self, x):
        output = []
        state = [None for _ in self.lstm]
        for i in range(x.shape[1]):
            out, state = self._forward_flow(x[:, i], state)
            output.append(out)
        return torch.stack([x for x in output], 1), state, output

    def _forward_flow(self, x, states):
        new_states = []
        for i, (layer, s) in enumerate(zip(self.lstm, states)):
            state = layer(x, s)
            x = state[0] if not i else x + state[0]
            new_states.append(state)
        mean = self._mean_layer(x)
        return mean, new_states

    def training_step(self, batch):
        data = batch
        x, y = data[:, :-1], data[:, 1:]
        y_pred, _, dist = self(x)
        loss = torch.clamp(WholeSeqence()(y_pred, y), max=1)
        self._report(
            "training",
            y_pred.detach().cpu().numpy(),
            y.cpu().numpy(),
            self._iteration_metrics,
        )
        return loss

    def validation_step(self, batch):
        data = batch
        x, y = data[:, :-1], data[:, 1:]
        y_pred, _, _ = self(x)
        self._report(
            "validation",
            y_pred.cpu().numpy(),
            y.cpu().numpy(),
            self._iteration_metrics,
        )

    def test_step(self, batch):
        data = batch
        x = data[:-1]
        for sample in x:
            input_seq, target_seq = sample[:50], sample[50:]
            output, _ = self._predict_step(input_seq, len(target_seq))
            self._report(
                "test",
                output,
                target_seq.cpu().numpy(),
                self._metrics,
            )

    def backtest(self, sample):
        sample_len = 500
        offset = sample[:-sample_len][:, -1].sum().item()
        sample = sample[-sample_len:]
        length = int(len(sample) // 2)
        input_seq, target_seq = sample[:length], sample[length:]
        output, sdvs = self._predict_step(input_seq, len(target_seq))
        sdvs = sdvs[:, 0]
        input_seq = input_seq[:, -1].numpy().cumsum() + offset
        target_seq = target_seq[:, -1].numpy().cumsum()
        output = output[:, -1].cumsum()
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

    def backtest_next_step(self, sample, seq, sample_orig):
        sample_size = 1000
        sample = sample[-sample_size:]
        input_seq, target_seq = sample[:-1], sample_orig[1:]
        output, _, _ = self.forward(input_seq.unsqueeze(0).cuda())
        output = output.detach().cpu().squeeze(0).numpy()
        # transform = RobustScaler().fit(seq.numpy())
        # output = transform.inverse_transform(output)
        output = sample_orig[-sample_size:-1, 0]*(output[:, 0]+1)
        # true_out = sample_orig[:-1, 0]*(seq[1:, 0]+1)
        # input_seq = self.revert_output(target_seq, np.zeros_like(output))
        offset = 100
        return sample_orig[-sample_size + offset+ 1:, 0], output[offset:], sample_orig[-sample_size + offset:-1, 0]

    def revert_output(self, input_seq, output):
        return (input_seq + output * 1e-7) / (1 - output)

    def _report(
        self,
        stage: str,
        output: torch.Tensor,
        target: torch.Tensor,
        metrics: typing.List[Metric],
    ):
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
        return optim.Adam(self.parameters(), lr=0.001)
