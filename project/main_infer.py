from data.dataloader import TimeSeriesDataset
from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite
from model.sequence import SequencePredictionModel

db = Sqlite("example2.db")
provider = Yahoo()

seq_length = 100
data = db.read({"Symbol": provider.tickers})
print(data.shape)

data = data.dropna(axis=1, thresh=len(data) - 5000)
data = data.dropna(axis=0)
print(data.shape)
cols = list(data.columns)
cols.remove("Symbol")
data = {key: subdata for key, subdata in data.groupby("Symbol")[cols]}
features = ["Open", "Close"]
backtest_dataset = TimeSeriesDataset(
    {"MSFT": data["MSFT"]}, len(data["MSFT"]) - 2, features
)
num_layers = 1
model = SequencePredictionModel.load_from_checkpoint(
    "checkpoints/00-val_loss3.24.ckpt",
    input_size=len(features),
    hidden_size=1024,
    output_size=len(features),
    num_layers=num_layers,
    seq_length=seq_length,
)
model.backtest(backtest_dataset[0][0])
