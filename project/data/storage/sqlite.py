import sqlite3

import pandas as pd


class Sqlite:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)

    def write(self, data: pd.DataFrame) -> None:
        data.to_sql("my_table", self._conn, if_exists="replace", index=False)

    def __del__(self):
        self._conn.close()
        super().__del__()
