from data.sources.yahoo import Yahoo
from data.storage.sqlite import Sqlite


db = Sqlite("example.db")
writer = Yahoo()
db.write(writer.history(writer.tickers[10]))
