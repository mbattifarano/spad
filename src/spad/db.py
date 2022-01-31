from pathlib import Path
from typing import Iterable, Type
from tqdm import tqdm
import osmnx as ox
import geopandas as gpd
import numpy as np
from toolz import curry, identity
from sqlalchemy.engine import Engine
from sqlalchemy import types as db_types
from sqlalchemy.dialects.postgresql import UUID
import pint

from logging import getLogger

from . import etl
from .common import SPADError
from .bounding_box import BoundingBox
from .osm import MAXSPEED_MPH_PATTERN

units = pint.UnitRegistry()
log = getLogger(__name__)

GPS_TABLE_NAME = "gps"
NODES_TABLE_NAME = "osmnx_nodes"
LINKS_TABLE_NAME = "osmnx_links"

LAT_LON_T = db_types.NUMERIC(15, 12)
OSMNX_TAG_T = db_types.ARRAY(db_types.TEXT)

TRAJECTORY_DBTYPES = dict(
    driver_id=UUID,
    shift_id=db_types.BIGINT,
    row_id=db_types.BIGINT,
    lat=LAT_LON_T,
    lon=LAT_LON_T,
    accuracy=db_types.REAL,
    speed=db_types.REAL,
    heading=db_types.REAL,
    altitude=db_types.REAL,
    activity_type=db_types.SMALLINT,
    activity_confidence=db_types.REAL,
    source=db_types.TEXT,
)

NODES_DBTYPES = dict(
    osmid=db_types.BIGINT,
    y=db_types.REAL,
    x=db_types.REAL,
    street_count=db_types.SMALLINT,
    lat=LAT_LON_T,
    lon=LAT_LON_T,
    highway=db_types.TEXT,
    ref=db_types.TEXT,
)

LINKS_DBTYPES = dict(
    u=db_types.BIGINT,
    v=db_types.BIGINT,
    key=db_types.INTEGER,
    osmid=db_types.BIGINT,
    oneway=db_types.BOOLEAN,
    lanes=db_types.REAL,
    maxspeed=db_types.REAL,
    lit=db_types.BOOLEAN,
    area=db_types.BOOLEAN,
    width=db_types.REAL,
)


@curry
def _cast_or_nan(cast, value):
    return cast(value) if not _isnan(value) else value


def _isnan(value):
    try:
        return np.isnan(value)
    except TypeError:
        return False


def _convert_maxspeed(s):
    try:
        unit = units(s)
    except pint.errors.PintError:
        log.warning(f"Could not convert maxspeed {s}; returning nan.")
        return np.nan
    if isinstance(unit, pint.Quantity):
        try:
            return unit.m_as("mph")
        except pint.errors.PintError:
            log.warning(f"Could not convert maxspeed {unit} to mph; returning nan.")
            return np.nan
    else:
        return unit


def _convert_width(s):
    try:
        unit = units(s.replace("'", "ft"))
    except pint.errors.PintError:
        log.warning(f"Could not convert width {s}; returning nan.")
        return np.nan
    if isinstance(unit, pint.Quantity):
        try:
            return unit.m_as("meter")
        except pint.errors.PintError:
            log.warning(f"Could not convert width {unit} to meters; returning nan.")
            return np.nan
    else:
        return unit


def _convert_osm_bool(s):
    return s == "yes"


LINK_COL_CONVERTERS = dict(
    lanes=_cast_or_nan(float),
    maxspeed=_cast_or_nan(_convert_maxspeed),
    lit=_cast_or_nan(_convert_osm_bool),
    area=_cast_or_nan(_convert_osm_bool),
    width=_cast_or_nan(_convert_width),
)


def import_trajectories(con: Engine, fnames: Iterable[Path]):
    for fname in tqdm(list(fnames)):
        import_trajectory_csv(con, fname)


def import_osmnx(con: Engine, bbox: BoundingBox, force=False):
    log.info("Building OSMnx graph.")
    g = bbox.to_osmnx()
    log.info("Building OSMnx GeoDataFrames.")
    nodes, links = ox.graph_to_gdfs(g)
    if_exists = "replace" if force else "fail"
    log.info("Importing links to the database.")
    osmnx_links_to_pg(links, con, if_exists)
    log.info("Importing nodes to the database.")
    osmnx_nodes_to_pg(nodes, con, if_exists)
    log.info("done.")


def osmnx_nodes_to_pg(nodes: gpd.GeoDataFrame, con: Engine, if_exists: str):
    nodes.reset_index(inplace=True)
    nodes.to_postgis(
        NODES_TABLE_NAME,
        con,
        index=False,
        if_exists=if_exists,
        dtype=NODES_DBTYPES,
    )


def _normalize_column_name(col):
    return col.replace(":", "__")


def osmnx_links_to_pg(links: gpd.GeoDataFrame, con: Engine, if_exists: str):
    links.reset_index(inplace=True)
    links.columns = links.columns.map(_normalize_column_name)
    for col, converter in LINK_COL_CONVERTERS.items():
        try:
            links[col] = links[col].apply(converter)
        except KeyError:
            log.warning(f"OSM tag {col} was not present in the OSM tag data.")
        except Exception as e:
            raise ETLError(f"Could not transform column {col}.") from e
    links.to_postgis(
        LINKS_TABLE_NAME,
        con,
        index=False,
        dtype=LINKS_DBTYPES,
        if_exists=if_exists,
    )


def is_container_type(obj) -> bool:
    return (
        isinstance(obj, list)
        or isinstance(obj, tuple)
        or isinstance(obj, set)
    )


def import_trajectory_csv(con: Engine, fname: Path):
    gdf = etl.read_csv(str(fname)).reset_index()
    gdf["source"] = fname.name
    gdf.to_postgis(
        GPS_TABLE_NAME,
        con,
        index=False,
        dtype=TRAJECTORY_DBTYPES,
        if_exists="append",
    )


class ETLError(SPADError):
    pass