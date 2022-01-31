"""Extract transform and load GPS trajectory data"""
from __future__ import annotations

import datetime as dt
from enum import Enum
from uuid import UUID

import geopandas as gpd
import numpy as np
import pandas as pd
from toolz import curry, pipe


def read_csv(fname: str, gap_threshold: float = 30.0) -> gpd.GeoDataFrame:
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
            Columns.shift_id.name: ToInt(default=-1),
            Columns.timestamp.name: from_ms_epoch,
            Columns.activity_type.name: ActivityType.as_int,
        },
    ).sort_index()
    df = pipe(
        df,
        add_row_id,
        compute_dt,
        compute_segment_ids(gap_threshold)
    )
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df[Columns.lon.name],
            df[Columns.lat.name],
            crs="WGS84"
        )
    )


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a unique integer id column to the dataframe."""
    df['row_id'] = np.arange(len(df))
    df.set_index('row_id', append=True, inplace=True)
    return df


def compute_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ping intervals in seconds for each pair of consecutive gps pings

    Pings are partitioned by driver and shift before dt is computed
    """
    df.reset_index(Columns.timestamp.name, inplace=True)
    df['dt'] = (
        df.groupby([Columns.driver_id.name, Columns.shift_id.name])
          .timestamp
          .diff()
          .dt
          .total_seconds()
    )
    df.set_index(Columns.timestamp.name, append=True, inplace=True)
    return df


@curry
def compute_segment_ids(threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    """Compute the segment ids for each driver and shift

    A segment of a driver's trajectory on a shift is defined as set of
    consecutive pings that are no more than threshold seconds apart.
    """
    df['segment_id'] = (df.groupby([Columns.driver_id.name,
                                    Columns.shift_id.name])
                          .dt.apply(_count_gaps(threshold))
                        )
    return df


@curry
def _count_gaps(gap_threshold: float, delta_t: pd.Series) -> pd.Series:
    return (~(delta_t <= gap_threshold)).cumsum()


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

    def __init__(self, vmin: float = 0, vmax: float = 100, default: float = 0):
        self.min = vmin
        self.max = vmax
        self.default = default

    def __call__(self, value):
        try:
            value = float(value)
        except ValueError:
            return self.default
        else:
            return (value - self.min) / (self.max - self.min)


class ToInt:
    """Convert input data to int returning a default value on failure"""

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
    def from_string(cls, name: str) -> ActivityType:
        """Return the ActivityType by name"""
        try:
            return cls[name]
        except KeyError:
            return cls.uncategorized

    @classmethod
    def as_int(cls, name: str) -> int:
        """Return the int value of the ActivityType by `name`"""
        return cls.from_string(name).value

    @classmethod
    def as_name(cls, value: int) -> str:
        """Return the name of the ActivityType by value"""
        return cls(value).name


EPOCH = dt.datetime(1970, 1, 1)


def from_ms_epoch(ms: int) -> dt.datetime:
    """Convert milliseconds since UNIX epoch to datetime"""
    return EPOCH + dt.timedelta(milliseconds=int(ms))
