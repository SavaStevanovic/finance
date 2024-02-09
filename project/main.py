import pandas as pd
from data.dataloader import TimeSeriesDataset
from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model.sequence import SequencePredictionModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

db = Sqlite("example2.db")
provider = Yahoo()

seq_length = 240
data = db.read({"Symbol": provider.tickers})
print(data.shape)

data = data.dropna(axis=1, thresh=len(data) - 5000)
data = data.dropna(axis=0)
print(data.shape)
cols = list(data.columns)
cols.remove("Symbol")
train_split = data[(data["Date"] < "2021-11-17 00:00:00-05:00")]
validation_split = data[
    (data["Date"] < "2022-11-17 00:00:00-05:00")
    & (data["Date"] >= "2021-11-17 00:00:00-05:00")
]
test_split = data[(data["Date"] >= "2022-11-17 00:00:00-05:00")]
training_data = {key: subdata for key, subdata in train_split.groupby("Symbol")[cols]}
val_data = {key: subdata for key, subdata in validation_split.groupby("Symbol")[cols]}
test_data = {key: subdata for key, subdata in test_split.groupby("Symbol")[cols]}
features = ["Open", "Close"]
train_dataset = TimeSeriesDataset(training_data, seq_length, features)
val_dataset = TimeSeriesDataset(val_data, seq_length, features)
test_dataset = TimeSeriesDataset(test_data, 100, features)
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
num_layers = 1
model = SequencePredictionModel(
    train_dataset[0][0].shape[1],
    1024,
    train_dataset[0][1].shape[1],
    num_layers,
    seq_length,
)
checkpoint_callback = ModelCheckpoint(
    monitor="validation/LossSeqence",
    dirpath="checkpoints/",
    filename="{epoch:02d}-val_loss{validation/LossSeqence:.2f}",
    auto_insert_metric_name=False,
)
trainer = pl.Trainer(
    max_epochs=50,
    gradient_clip_val=1,
    gradient_clip_algorithm="value",
    callbacks=[checkpoint_callback],
)
trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, test_dataloader)
