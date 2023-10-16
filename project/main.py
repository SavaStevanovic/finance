from data.sources.data_source import Source
from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite
from tqdm import tqdm

db = Sqlite("example.db")
provider = Yahoo()
writer = Source(provider)
for ticket in tqdm(provider.tickers):
    db.write(writer.history(ticket))
