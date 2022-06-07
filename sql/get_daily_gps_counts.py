import pandas as pd
import psycopg2 as pg
import time


QUERY = """
select
    date_trunc('hour', (gps.timestamp at time zone 'UTC' at time zone 'America/New_York')) as day_hour,
    count(*) as n_pings
from gps_still gps
group by 1
"""

print("Running Query")
with pg.connect() as conn:
    t0 = time.time()
    df = pd.read_sql(QUERY, conn)
    print(f"Query completed in {time.time()-t0}.")

df.to_parquet("day_hour_pings.parquet")

