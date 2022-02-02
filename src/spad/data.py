from re import L
from typing import Iterable, Tuple
import datetime as dt
import uuid
from matplotlib.pyplot import table
import psycopg2 as pg
from psycopg2.extras import NamedTupleCursor
import geopandas as gpd
import pandas as pd
import pint
import pint_pandas
import numpy as np
import shapely
import networkx as nx
import osmnx as ox
from sqlalchemy import sql, func as sql_fn
import logging
from . import db
from .common import SPADError, to_travel_time, units
from textwrap import dedent


log = logging.getLogger(__name__)
LINK_SPEED = "travel_speed"
LINK_TRAVEL_TIME = "travel_time"
MAX_SPEED_LIMIT = 70.0 * units.mph
TRAVEL_TIME_UNITS = "minutes"
MIN_ACCURACY = 1.0


def driver_segments(
    conn,
    threshold: dt.timedelta,
    min_duration: dt.timedelta = None,
    start_at_driver: uuid.UUID = None,
) -> Iterable[gpd.GeoDataFrame]:
    cursor_kws = dict(
        name=f"driver_segments_cursor-{uuid.uuid4()}",
        cursor_factory=NamedTupleCursor,
        withhold=True,
    )
    log.info(f"Using server-side cursor {cursor_kws['name']}.")
    cursor = conn.cursor(**cursor_kws)
    try:
        cursor.execute(_gps_segments_query(start_at_driver))
        records = []
        current_driver = None
        for row in cursor:
            dt = _compute_dt(records, row)
            if dt > threshold or _is_new_driver(current_driver, row.driver_id):
                if _exceeds_duration(min_duration, records):
                    yield _to_geo_dataframe(records)
                current_driver = row.driver_id
                records.clear()
            records.append(row)
        if _exceeds_duration(min_duration, records):
            yield _to_geo_dataframe(records)
    finally:
        cursor.close()


def _is_new_driver(current_driver, this_driver):
    return (this_driver != current_driver) and (current_driver is not None)


def get_osmnx_table(conn, table_name: str, geom_wkb: str = None, **kws):
    params = [geom_wkb] if geom_wkb else None
    return gpd.GeoDataFrame.from_postgis(
        osmnx_sql(table_name, geom_wkb),
        conn,
        geom_col="geometry",
        params=params,
        **kws,
    )


def osmnx_sql(table_name: str, geom_wkb: str = None) -> str:
    if table_name == db.NODES_TABLE_NAME:
        stmt = dedent(
            f"""
        SELECT * FROM {table_name}
        """
        )
    elif table_name == db.LINKS_TABLE_NAME:
        stmt = dedent(
            f"""
            with
                highway_avg_speed as (
                    select
                        highway,
                        sum(maxspeed * n_links) / sum(n_links) as maxspeed
                    from osmnx_default_maxspeed
                    group by 1
                )
            select
                l.*,
                coalesce(l.maxspeed,
                         ds.maxspeed,
                         ha.maxspeed,
                         15.0) as {LINK_SPEED}
            from {table_name} l
            left join pa_municipalities pm
                on st_intersects(st_centroid(l.geometry), pm.geom)
            left join osmnx_default_maxspeed ds
                on pm.fed_aid_ur = ds.fed_urban_level
                   and l.highway = ds.highway
            left join highway_avg_speed ha
                on l.highway = ha.highway
        """
        )
    else:
        raise DataError(
            f"Unrecognized table name: {table_name}. "
            "Must be one of [{db.NODES_TABLE_NAME}, {db.LINKS_TABLE_NAME}]"
        )
    if geom_wkb:
        stmt += "WHERE ST_Intersects(geometry, %s)"
    log.debug(f"OSMNX SQL for {table_name}: {stmt}")
    return stmt


def get_osmnx_gdfs(conn, geom_wkb: str = None):
    links = get_osmnx_table(
        conn, db.LINKS_TABLE_NAME, geom_wkb, index_col=["u", "v", "key"]
    )
    nodes = get_osmnx_table(
        conn, db.NODES_TABLE_NAME, geom_wkb, index_col=["osmid"]
    )
    return nodes, links


def get_osmnx_network(conn, geom_wkb: str = None):
    log.info("Retrieving OSMNX network data.")
    nodes, links = get_osmnx_gdfs(conn, geom_wkb)
    links[LINK_TRAVEL_TIME] = to_travel_time(
        links.length.astype("pint[meter]"),
        links[LINK_SPEED].astype("pint[mph]"),
    ).pint.m_as(TRAVEL_TIME_UNITS)
    log.info("Forming OSMNX graph.")
    g = ox.graph_from_gdfs(nodes, links)
    log.info("done.")
    return g, links


def get_osmnx_subnetwork(
    conn, trajectory: gpd.GeoDataFrame, buffer: float = 1000.0
) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    hull = _get_buffered_convex_hull_wkb(trajectory, buffer)
    return get_osmnx_network(conn, hull)


def _get_buffered_convex_hull_wkb(trajectory: gpd.GeoDataFrame, buffer: float):
    utm = trajectory.estimate_utm_crs()
    hull = trajectory.set_geometry(
        trajectory.to_crs(utm).buffer(buffer)
    ).unary_union.convex_hull
    return shapely.wkb.dumps(hull, hex=True, srid=utm.to_epsg())


def _exceeds_duration(threshold, records):
    duration = records[-1].timestamp - records[0].timestamp
    return duration >= (threshold or dt.timedelta(0))


def _to_geo_dataframe(records, geometry_column="geometry"):
    df = pd.DataFrame(records)
    df.index.rename("ping_order", inplace=True)
    df.set_index("id", inplace=True, append=True)
    _repair_accuracy(df)
    df[geometry_column] = geoms = df[geometry_column].apply(_to_geom)
    return gpd.GeoDataFrame(df, crs=_get_srid(geoms), geometry=geometry_column)


def _repair_accuracy(df):
    max_accuracy = df.accuracy.max()
    df.accuracy.fillna(max_accuracy, inplace=True)
    df.loc[df.accuracy < 0.0, "accuracy"] = max_accuracy
    df.accuracy.clip(lower=MIN_ACCURACY, inplace=True)


def _get_srid(geoms):
    srid = shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom)
    # if no defined SRID in geodatabase, returns SRID of 0
    if srid != 0:
        return "epsg:{}".format(srid)


def _to_geom(wkb_value):
    return shapely.wkb.loads(str(wkb_value), hex=True)


def _compute_dt(records, row) -> float:
    if records:
        return row.timestamp - records[-1].timestamp
    else:
        return dt.timedelta(0)


def _gps_segments_query(start_at_driver: uuid.UUID = None) -> str:
    stmt = f"SELECT * FROM {db.GPS_TABLE_NAME}"
    if start_at_driver is not None:
        stmt += f" WHERE driver_id >= '{start_at_driver}'"
    stmt += " ORDER BY driver_id, timestamp"
    log.info(f"GPS segments query: {stmt}")
    return stmt


class DataError(SPADError):
    pass
