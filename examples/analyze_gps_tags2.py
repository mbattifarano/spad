import asyncio
from multiprocessing import Semaphore
import sys
import time
import datetime as dt
import pandas as pd
from typing import Iterable, NamedTuple, List
import aiopg
import psycopg2 as pg
import click
from tqdm.asyncio import tqdm_asyncio


@click.command()
@click.option("--radius", type=float, default=75.0)
@click.option("--outfile", "-o")
@click.option("--n-threads", type=int, default=16)
def analyze_gps_tags(radius, outfile, n_threads):
    idx = ParamIndex.to_index()
    t0 = time.time()
    with pg.connect() as conn:
        ping_tag_groups = pd.read_sql(PING_GROUPS_SQL, conn, index_col=idx)
    click.echo(f"Retrieved ping tag groups in {time.time()-t0:0.2f}s.")
    t1 = time.time()
    all_counts = asyncio.run(
        get_many_ping_tags(
            (
                ParamIndex.from_row(radius, _idx)
                for _idx, _ in ping_tag_groups.iterrows()
            ),
            n_threads=n_threads,
            total=len(ping_tag_groups),
        )
    )
    click.echo(
        f"Retrieved all ping tag interactions in {time.time()-t1:0.2f}s."
    )
    all_counts = pd.DataFrame(
        all_counts, columns=ParamIndex._fields + COUNT_PING_TAGS_COLUMNS
    )
    all_counts.set_index(idx, inplace=True)
    result = ping_tag_groups.join(all_counts)
    result.set_index("radius", append=True, inplace=True)
    click.echo(f"Query completed in {time.time()-t0:0.2f}s.")
    if outfile:
        click.echo(f"Saving results to {outfile}.")
        result.to_parquet(outfile)


class ParamIndex(NamedTuple):
    year_month: dt.datetime
    hour_of_day: int
    tag_name: str
    tag_value: str
    radius: float

    @classmethod
    def from_row(cls, radius: float, row: tuple):
        return cls(*row, radius)

    @classmethod
    def to_index(cls) -> List[str]:
        return list(cls._fields[:4])


async def get_many_ping_tags(
    params: Iterable[ParamIndex], n_threads: int, total: int
):
    async with aiopg.create_pool(maxsize=n_threads, timeout=None) as pool:
        ret = await tqdm_asyncio.gather(
            *[get_ping_tags(pool, p) for p in params], total=total
        )
        return ret


QUERY_TIMEOUT = dt.timedelta(hours=1).total_seconds()


async def get_ping_tags(pool, params: ParamIndex):
    with (await pool.cursor(timeout=None)) as cur:
        await cur.execute(COUNT_PING_TAGS_SQL, params._asdict())
        row = await cur.fetchone()
        return tuple(params) + row


PING_GROUPS_SQL = """
select *
from (
        select year_month, hour_of_day, count(*) as n_pings
        from gps_still
        group by 1, 2
    ) ps,
    (
        select et.tag_name, et.tag_value, count(*) as n_tags
        from relevant_osm_entity_tags et
        group by 1, 2
    ) tags
"""

COUNT_PING_TAGS_COLUMNS = ("n_interactions",)
COUNT_PING_TAGS_SQL = """
select count(*) as n_interactions
from gps_still gpss
join relevant_osm_entity_tags et
on st_dwithin(st_transform(gpss.geometry, 32617), et.geometry, %(radius)s)
where gpss.year_month = %(year_month)s
    and gpss.hour_of_day = %(hour_of_day)s
    and et.tag_name = %(tag_name)s
    and et.tag_value = %(tag_value)s
"""


if __name__ == "__main__":
    analyze_gps_tags()
