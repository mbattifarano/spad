from time import time
from typing import Tuple
import pandas as pd
import psycopg2 as pg
import click
import logging

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s@%(lineno)d %(funcName)s: %(message)s",  # noqa
)

log = logging.getLogger()

QUERY = """
select
    date_trunc('month', gps.timestamp)::date as year_month,
    gps.activity_type,
    et.tag_name, et.tag_value, count(*) as tag_count
from  gps
join osm_entity_tags et
on st_dwithin(st_transform(gps.geometry, 32617), et.geometry, %(radius)s)
where gps.id between %(start_id)s and %(end_id)s
group by 1, 2, 3, 4
order by 1, 2, 3, 4
"""


def id_intervals(chunksize: int, stop: int, start: int = 1) -> Tuple[int, int]:
    start_id = start
    while start_id <= stop:
        end_id = start_id + chunksize - 1
        yield start_id, end_id
        start_id = end_id + 1


def get_tags(conn, start_id: int, end_id: int, radius: float):
    return pd.read_sql(
        QUERY,
        conn,
        index_col=["year_month", "activity_type", "tag_name", "tag_value"],
        params=dict(
            start_id=start_id,
            end_id=end_id,
            radius=radius,
        ),
    )


def join_tag_count(df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    return (
        new_df.join(df, how="outer", rsuffix="_r")
        .sum(axis=1, skipna=True)
        .rename("tag_count")
    )


def all_tag_counts(conn, start, stop, chunksize, radius, outfile) -> pd.Series:
    df = None
    start_ends = list(id_intervals(chunksize, stop, start))
    for start_id, end_id in tqdm(start_ends):
        new_df = get_tags(conn, start_id, end_id, radius)
        if df is None:
            df = new_df.tag_count
        else:
            df = join_tag_count(df, new_df)
        if outfile is not None:
            df.to_frame().to_parquet(outfile)
    return df


@click.command()
@click.option("--start", type=int, default=1)
@click.option("--chunksize", type=int, default=10000)
@click.option("--stop", type=int, required=True)
@click.option(
    "--radius",
    type=float,
    default=400.0,
    help="Radius in meters around the gps ping to gather tags.",
)
@click.option("-o", "--outfile")
def get_gps_tags(start, chunksize, stop, radius, outfile):
    with pg.connect() as conn:
        df = all_tag_counts(conn, start, stop, chunksize, radius, outfile)
    if outfile is not None:
        log.info(f"Saving {len(df)} gps tag summaries to {outfile}.")
        df.to_frame().to_parquet(outfile)
    log.info("done.")


if __name__ == "__main__":
    get_gps_tags()
