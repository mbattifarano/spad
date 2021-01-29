from __future__ import annotations

import datetime as dt
from enum import Enum
from uuid import UUID

import geopandas as gpd
import pandas as pd


def read_csv(fname: str) -> pd.DataFrame:
    """Read GPS trajectory data from a csv file"""
    df = pd.read_csv(
        fname,
        index_col=(
            Columns.driver_id.name,
            Columns.shift_id.name,
            Columns.timestamp.name
        ),
        converters={
            Columns.driver_id.name: UUID,
            Columns.shift_id.name: ToInt(default=0),
            Columns.timestamp.name: from_ms_epoch,
            Columns.activity_type.name: ActivityType.as_int,
            Columns.activity_confidence.name: Normalizer(),
        },
    ).sort_index()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df[Columns.lon.name],
            df[Columns.lat.name],
            crs="WGS84"
        )
    )


class Columns(Enum):
    """Column names and positions in the csv data"""

    driver_id = 0
    shift_id = 1
    timestamp = 2
    lat = 3
    lon = 4
    accuracy = 5
    speed = 6
    heading = 7
    altitude = 8
    activity_type = 9
    activity_confidence = 10
    odometer = 11


class Normalizer:
    """Normalize data within predefined bounds."""

    def __init__(self, min: float = 0, max: float = 100, default: float = 0):
        self.min = min
        self.max = max
        self.default = default

    def __call__(self, value):
        try:
            value = float(value)
        except ValueError:
            return self.default
        else:
            return (value - self.min) / (self.max - self.min)


class ToInt:
    def __init__(self, default: int):
        self.default = int(default)

    def __call__(self, value):
        try:
            return int(value)
        except ValueError:
            return self.default


class ActivityType(Enum):
    """Represent the activity types in the GPS trajectory data"""

    uncategorized = 1
    unknown = 2
    still = 3
    on_foot = 4
    walking = 5
    running = 6
    on_bicycle = 7
    in_vehicle = 8

    @classmethod
    def from_string(cls, value: str) -> ActivityType:
        try:
            return cls[value]
        except KeyError:
            return cls.uncategorized

    @classmethod
    def as_int(cls, value: str) -> int:
        return cls.from_string(value).value

    @classmethod
    def as_name(cls, value: int) -> str:
        return cls(value).name


EPOCH = dt.datetime(1970, 1, 1)


def from_ms_epoch(ms: int) -> dt.datetime:
    """Convert milliseconds since UNIX epoch to datetime"""
    return EPOCH + dt.timedelta(milliseconds=int(ms))
