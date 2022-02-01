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
from .common import to_travel_time


log = logging.getLogger(__name__)


units = pint.UnitRegistry()


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
    stmt = f"SELECT * FROM {table_name}"
    if geom_wkb:
        stmt += f" WHERE ST_Intersects(geometry, %s)"
    log.info(f"OSMNX SQL for {table_name}: {stmt}")
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
    log.info("Forming OSMNX graph.")
    g = ox.graph_from_gdfs(nodes, links)
    log.info("done.")
    return g, links


ROAD_CLASS_SPEEDS = {
    "motorway": 70.0,
    "motorway_link": 70.0,
    "trunk": 65.0,
    "tunk_link": 55.0,
    "primary": 55.0,
    "primary_link": 50.0,
    "secondary": 45.0,
    "secondary_link": 45.0,
    "road": 45.0,
    "tertiary": 35.0,
    "tertiary_link": 35.0,
    "unclassified": 35.0,
    "residential": 25.0,
    "living_street": 15.0,
    "alley": 15.0,
    "service": 15.0,
}

SERVICE_SPEEDS = {
    "alley": 15.0,
    "driveway": 10.0,
    "parking_aisle": 10.0,
    "aisle": 10.0,
    "drive-through": 10.0,
    "Service_Road": 15.0,
}

DEFAULT_ROAD = 25.0


def infer_speed(link_row: pd.Series):
    if not np.isnan(link_row.maxspeed):
        return link_row.maxspeed
    service_speed = SERVICE_SPEEDS.get(link_row.service)
    if service_speed is not None:
        return service_speed
    return ROAD_CLASS_SPEEDS.get(link_row.highway, DEFAULT_ROAD)


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


LINK_SPEED = "speed"
LINK_TRAVEL_TIME = "travel_time"
MAX_SPEED_LIMIT = 70.0 * units.mph
TRAVEL_TIME_UNITS = "minutes"


def _to_geo_dataframe(records, geometry_column="geometry"):
    df = pd.DataFrame(records)
    df.index.rename("ping_order", inplace=True)
    df.set_index("id", inplace=True, append=True)
    _repair_accuracy(df)
    df[LINK_SPEED] = df.apply(infer_speed, axis=1)
    df[LINK_TRAVEL_TIME] = to_travel_time(
        df.length.astype("pint[meter]"), df[LINK_SPEED].astype("pint[mph]")
    ).pint.m_as(TRAVEL_TIME_UNITS)
    df[geometry_column] = geoms = df[geometry_column].apply(_to_geom)
    return gpd.GeoDataFrame(df, crs=_get_srid(geoms), geometry=geometry_column)


def _repair_accuracy(df):
    df.accuracy.fillna(df.accuracy.max(), inplace=True)
    df.accuracy.clip(lower=1.0, inplace=True)


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
