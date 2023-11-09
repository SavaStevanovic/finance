import pandas as pd
from data.dataloader import TimeSeriesDataset
from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model.sequence import SequencePredictionModel
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint

db = Sqlite("example.db")
provider = Yahoo()

seq_length = 100
data = db.read({"Symbol": provider.tickers})
print(data.shape)

data = data.dropna(axis=1, thresh=len(data) - 5000)
data = data.dropna(axis=0)
print(data.shape)
cols = list(data.columns)
cols.remove("Symbol")
groups = {key: subdata for key, subdata in data.groupby("Symbol")[cols]}
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
checkpoint_callback = ModelCheckpoint(
    monitor="validation/LossSeqence",
    dirpath="checkpoints/",
    filename="{epoch:02d}-val_loss{validation/LossSeqence:.2f}",
    auto_insert_metric_name=False,
)
trainer = pl.Trainer(
    max_epochs=20,
    gradient_clip_val=1,
    gradient_clip_algorithm="value",
    callbacks=[checkpoint_callback],
)
# trainer.fit(model, train_dataloader, val_dataloader)

model = SequencePredictionModel.load_from_checkpoint(
    "checkpoints/11-val_loss0.31.ckpt",
    input_size=train_dataset[0][0].shape[1],
    hidden_size=1024,
    output_size=train_dataset[0][1].shape[1],
    num_layers=1,
    seq_length=seq_length,
)
trainer.test(model, test_dataloader)
