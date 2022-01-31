"""OpenStreetMaps parsing and feature engineering"""
from collections import defaultdict
import re
from typing import Iterable, Mapping, Tuple
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx

from .common import SPADError


# Modify the `drive_service` OSMNX filter to include private roads
OSMNX_NETWORK_FILTER = (
    '["highway"]["area"!~"yes"]'
    '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor'
    "|cycleway|elevator|escalator|footway|path|pedestrian|planned|platform"
    '|proposed|raceway|steps|track"]'
    '["motor_vehicle"!~"no"]["motorcar"!~"no"]'
    '["service"!~"emergency_access"]'
)
# Bring in more tags
EXTRA_WAY_TAGS = [
    "abutters",
    "incline",
    "lit",
    "bus_bay",
    "parking:lane",
    "traffic_calming",
    "bicycle",
]

TAG_COL = "tag"

ox.settings.useful_tags_way.extend(
    set(EXTRA_WAY_TAGS) - set(ox.settings.useful_tags_way)
)

FEATURE_TAGS = [
    "amenity",
    "building",
    "craft",
    "emergency",
    "healthcare",
    "leisure",
    "landuse",
    "office",
    "public_transport",
    "shop",
    "sport",
    "tourism",
    "route",
]

EMPTY = "unspecified"
NO = "no"
DEFAULT_SPEED_LIMIT = "25 mph"
MAXSPEED_MPH_PATTERN = re.compile(r"(\d+)\s*mph")

TAG_DEFAULTS = {
    "service": EMPTY,
    "lanes": "2",
    "ref": EMPTY,
    "maxspeed": "25 mph",
    "access": EMPTY,
    "bridge": NO,
    "tunnel": NO,
    "bicycle": EMPTY,
    "name": EMPTY,
}

FEATURE_DTYPES = {
    "lanes": np.uint8,
    "maxspeed": np.uint8,
}


def to_numeric_maxspeed(s):
    return int(MAXSPEED_MPH_PATTERN.match(s).group(1))


def to_tuple(scalar_or_list):
    if isinstance(scalar_or_list, list):
        scalar_or_list = [scalar_or_list]
    return tuple(scalar_or_list)


FEATURE_TRANSFORMS = {
    "maxspeed": to_numeric_maxspeed,
    "osmid": to_tuple,
    "highway": to_tuple,
}


def get_network_features(
    bbox, link_buffer_meters: float = 100.0
) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    g = bbox.to_osmnx(utm=True)
    _, links = ox.graph_to_gdfs(g)
    buffered_links = links.set_geometry(links.buffer(link_buffer_meters))
    all_tags, places = get_osm_tags(bbox)
    link_index = links.index.names
    link_tags = (
        buffered_links.sjoin(places, how="inner", predicate="intersects")
        .groupby(link_index)[all_tags]
        .sum()
    )
    highways = to_count_features(
        links.highway.explode().apply("highway:{}".format).to_frame(),
        "highway",
    )
    return g, transform(impute(links)).join(link_tags).join(highways).fillna(0)


def get_osm_tags(bbox, dtype=np.uint8):
    osm_tags = get_tags_within(bbox, FEATURE_TAGS)
    tag_groups = get_tag_groups(osm_tags)
    all_tags = list(flatten(tag_groups.values()))
    places = to_count_features(osm_tags, dtype=dtype).to_crs(bbox.get_utm())
    for tagname, tags in tag_groups.items():
        places[tagname] = places[tags].sum(axis=1).astype(dtype)
    return all_tags, places


def impute(links: pd.DataFrame) -> pd.DataFrame:
    is_imputed = links[TAG_DEFAULTS.keys()].isna()
    is_imputed.columns = is_imputed.columns.map("{}_imputed".format)
    return links.fillna(TAG_DEFAULTS).join(is_imputed)


def transform(links: pd.DataFrame) -> pd.DataFrame:
    return links.apply(FEATURE_TRANSFORMS).astype(FEATURE_DTYPES)


def flatten(xs: Iterable[Iterable]) -> Iterable:
    for ys in xs:
        yield from ys


def to_count_features(
    gdf: gpd.GeoDataFrame, col: str = TAG_COL, dtype=np.uint8
) -> gpd.GeoDataFrame:
    index = list(gdf.index.names)
    df = gdf[[col]]
    _value = "value"
    df[_value] = 1
    df.reset_index().pivot(index=index, columns=col, value=_value,).fillna(
        0
    ).astype(dtype)
    return gpd.GeoDataFrame(df, geometry=gdf.geometry)


def get_tag_groups(df: pd.DataFrame) -> Mapping[str, str]:
    tags = df[TAG_COL]
    tg = defaultdict(list)
    for tag in tags:
        tagname, _ = split_tag(tag)
        tg[tagname].append(tag)
    return dict(tg)


def split_tag(tag: str) -> Tuple[str, str]:
    return tag.split(":")


def get_tags_within(bbox, tags: Iterable[str]) -> gpd.GeoDataFrame:
    gdfs = {}
    for tag in tags:
        gdf = get_tag_within(bbox, tag)
        if gdf is not None:
            gdfs[tag] = gdf
    if not gdfs:
        raise OSMError("No tags were found within the bounding box.")
    return pd.concat(gdfs, names=["tagname"])


def get_columns(
    gdf: gpd.GeoDataFrame, cols: Iterable[str]
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(gdf[cols], geometry=gdf.geometry)


def get_tag_within(bbox, tag: str) -> gpd.GeoDataFrame:
    gdf = ox.geometries_from_bbox(
        *bbox.to_cardinal_directions(),
        {tag: True},
    )
    if tag in gdf.columns:
        gdf[tag] = gdf[tag].apply((tag + ":{}").format)
        return gdf


class OSMError(SPADError):
    pass
