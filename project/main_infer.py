from data.dataloader import TimeSeriesDataset
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
test_split = data[(data["Date"] >= "2022-11-17 00:00:00-05:00")]
print(data.shape)
cols = list(data.columns)
cols.remove("Symbol")
data = {key: subdata for key, subdata in data.groupby("Symbol")[cols]}
features = ["Open", "Close"]
backtest_dataset = TimeSeriesDataset(data, 1000, features, True)
num_layers = 1
model = SequencePredictionModel.load_from_checkpoint(
    "checkpoints/08-val_loss4.83.ckpt",
    input_size=backtest_dataset[0][0].shape[1],
    hidden_size=1024,
    output_size=backtest_dataset[0][0].shape[1],
    num_layers=num_layers,
    seq_length=seq_length,
)
model.backtest(backtest_dataset[0][0])
model.backtest_next_step(backtest_dataset[0][0])
