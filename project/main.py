import multiprocessing
import os

import pandas as pd
from tqdm import tqdm
from data.dataloader import TimeSeriesDataset
from data.sources.data_source import Source
from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import loggers as pl_loggers
from model.sequence import SequencePredictionModel
from sklearn.model_selection import train_test_split

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
db = Sqlite("example.db")
provider = Yahoo()
writer = Source(provider)
for ticket in tqdm(provider.tickers):
    db.write(writer.history(ticket))
q = {"Symbol": "A"}
print(q)
s = db.read(q)
# print(s)

seq_length = 100
data = db.read({"Symbol": provider.tickers})
print(data.shape)

data = data.dropna(axis=1, thresh=len(data) - 5000)
data = data.dropna(axis=0)
print(data.shape)
cols = list(data.columns)
cols.remove("Symbol")
groups = {
    key: subdata for key, subdata in data.groupby("Symbol")[cols] if len(key) == 4
}
s = pd.Series(groups)
training_data, data = [
    i.to_dict() for i in train_test_split(s, train_size=0.7, random_state=1)
]
s = pd.Series(data)

val_data, test_data = [
    i.to_dict() for i in train_test_split(s, train_size=0.5, random_state=1)
]
train_dataset = TimeSeriesDataset(training_data, seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, 100)
print(train_dataset[0])
print(s.head())

train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=23, pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=23,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=23,
)

model = SequencePredictionModel(
    train_dataset[0][0].shape[1],
    1024,
    train_dataset[0][1].shape[1],
    1,
    seq_length,
)
trainer = pl.Trainer(
    max_epochs=30, gradient_clip_val=1, gradient_clip_algorithm="value"
)

trainer.fit(model, train_dataloader, val_dataloader)

# model = SequencePredictionModel.load_from_checkpoint(
#     "lightning_logs/version_121/checkpoints/epoch=15-step=15488.ckpt",
#     input_size=train_dataset[0][0].shape[1],
#     hidden_size=1024,
#     output_size=train_dataset[0][1].shape[1],
#     num_layers=1,
#     seq_length=seq_length,
# )
trainer.test(model, test_dataloader)
s = test_data["CSCO"]
start = 5000
seq = TimeSeriesDataset.preprocess(s.iloc[start : start + 100])
print(model.predict_step(seq, 100))
print(s["Open"].values.tolist()[:200])
import numpy as np

t = s["Open"].values.tolist()[start + 100 : start + 200]
p = model.predict_step(seq, 100)
print(np.abs(np.array(t) - np.array(p)) / np.array(t))
