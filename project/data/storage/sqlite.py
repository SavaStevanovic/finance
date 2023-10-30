import sqlite3

import pandas as pd


class Sqlite:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)

    def write(self, data: pd.DataFrame) -> None:
        data.to_sql("my_table", self._conn, if_exists="append")

    def read(self, filter: dict) -> pd.DataFrame:
        query = "select * from my_table"
        if filter:
            search = []
            for k, v in filter.items():
                v = [f"'{e}'" for e in v]
                search.append(f"{k} in (" + ", ".join(v) + ")")
            query += f" where " + " and ".join(search)
        return pd.read_sql_query(query, self._conn)

    def __del__(self):
        self._conn.close()
