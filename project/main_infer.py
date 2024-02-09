import torch
from data.dataloader import TimeSeriesDatasetInference
from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite
from model.sequence import SequencePredictionModel

db = Sqlite("example2.db")
provider = Yahoo()

seq_length = 100
data = db.read({})
print(data.shape)

data = data.dropna(axis=1, thresh=len(data) - 5000)
data = data.dropna(axis=0)
print(data.shape)
cols = list(data.columns)
cols.remove("Symbol")
data = {key: subdata for key, subdata in data.groupby("Symbol")[cols]}
features = ["Close"]
backtest_dataset = TimeSeriesDatasetInference(data, 1000, features)
num_layers = 1
model = SequencePredictionModel.load_from_checkpoint(
    "checkpoints/01-val_loss0.75.ckpt",
    input_size=backtest_dataset[0][1].shape[1],
    hidden_size=1024,
    output_size=backtest_dataset[0][1].shape[1],
    num_layers=num_layers,
    seq_length=seq_length,
).cuda()
symbol_weights = {
    "MMM": 2.41,
    "AXP": 3.02,
    "AMGN": 5.48,
    "AAPL": 2.84,
    "BA": 3.36,
    "CAT": 4.52,
    "CVX": 3.50,
    "CSCO": 0.96,
    "KO": 1.22,
    "DIS": 1.89,
    "DOW": 0.98,
    "GS": 7.36,
    "HD": 6.27,
    "HON": 4.17,
    "IBM": 2.86,
    "INTC": 0.57,
    "JNJ": 3.43,
    "JPM": 2.61,
    "MCD": 5.24,
    "MRK": 2.10,
    "MSFT": 4.88,
    "NKE": 2.13,
    "PG": 2.86,
    "CRM": 2.82,
    "TRV": 3.62,
    "UNH": 10.29,
    "VZ": 0.73,
    "V": 4.16,
    "WBA": 0.79,
    "WMT": 2.94,
}
symbol_weights_total = sum(symbol_weights.values())
symbol_weights = {k: v / symbol_weights_total for k, v in symbol_weights.items()}

# model.backtest(backtest_dataset[0][0])
market_data = {}
for market, data, seq, data_orig in backtest_dataset:
    market_data[market] = model.backtest_next_step(data, seq, data_orig)


def sum_symbol_data(symbol_weights, market_data, id):
    return sum(symbol_weights[k] * x[id] for k, x in market_data.items())


target_seq = sum_symbol_data(symbol_weights, market_data, 0)
output = sum_symbol_data(symbol_weights, market_data, 1)
input_seq = sum_symbol_data(symbol_weights, market_data, 2)

import matplotlib.pyplot as plt
import numpy as np

plt.xlabel("Time")
plt.ylabel("Value")
print(plt.rcParamsDefault["figure.figsize"])
plt.plot(target_seq, label=f"target", linewidth=4)
plt.plot(output, label=f"output", linewidth=3.0)
target_pred = np.absolute(target_seq - output)
plt.plot(target_pred, label=f"target_seq - output", linewidth=1)
same_seq = np.absolute(target_seq - input_seq)
plt.plot(same_seq, label=f"target_seq - input_seq", linewidth=1)
plt.legend()

plt.savefig("msft_single.png")
print(f"pred_value {target_pred.sum()}")
print(f"input_value {same_seq.sum()}")


target_seq, output, input_seq = model.backtest_next_step(
    sum(symbol_weights[market] * data for market, data, _, _ in backtest_dataset),
    sum(symbol_weights[market] * data for market, _, data, _ in backtest_dataset),
    sum(symbol_weights[market] * data for market, _, _, data in backtest_dataset),
)
plt.clf()
plt.xlabel("Time")
plt.ylabel("Value")
print(plt.rcParamsDefault["figure.figsize"])
plt.plot(target_seq, label=f"target", linewidth=4)
plt.plot(output, label=f"output", linewidth=3.0)
target_pred = np.absolute(target_seq - output)
plt.plot(target_pred, label=f"target_seq - output", linewidth=1)
same_seq = np.absolute(target_seq - input_seq)
plt.plot(same_seq, label=f"target_seq - input_seq", linewidth=1)
plt.legend()

plt.savefig("msft_single_total.png")
print(f"pred_value {target_pred.sum()}")
print(f"input_value {same_seq.sum()}")
