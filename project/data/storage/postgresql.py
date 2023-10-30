import pandas as pd
import psycopg2


class PostgreSQLDataWriter:
    def __init__(self, host, database, user, password):
        self._con = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
        )

    def write(self, data: pd.DataFrame) -> None:
        data.to_sql("my_table", self._conn, if_exists="replace")

    def __del__(self):
        self._conn.close()
        super().__del__()
