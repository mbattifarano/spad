from __future__ import annotations

from typing import NamedTuple

import geopandas as gpd
import networkx as nx
import osmnx as ox

from pyproj import CRS
from pyproj.database import query_utm_crs_info, CRSInfo
from pyproj.aoi import AreaOfInterest


WGS84 = "WGS84"


class BoundingBox(NamedTuple):
    """Represents a geospatial bounding box"""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def north(self) -> float:
        """Northern extent of the bounding box"""
        return self.max_lat

    @property
    def south(self) -> float:
        """Southern extent of the bounding box"""
        return self.min_lat

    @property
    def east(self) -> float:
        """Eastern extent of the bounding box"""
        return self.max_lon

    @property
    def west(self) -> float:
        """Western extent of the bounding box"""
        return self.min_lon

    @staticmethod
    def from_geodataframe(df: gpd.GeoDataFrame, buffer: float = 0.0) \
            -> BoundingBox:
        """Create a bounding box for the spatial data in a geodataframe"""
        minx, miny, maxx, maxy = df.geometry.total_bounds
        return BoundingBox(
            min_lon=minx - buffer,
            min_lat=miny - buffer,
            max_lon=maxx + buffer,
            max_lat=maxy + buffer,
        )

    def to_osmnx(self) -> nx.MultiDiGraph:
        """Create an osmnx network from the bounding box"""
        return ox.graph_from_bbox(
            self.north,
            self.south,
            self.east,
            self.west,
            network_type="drive",
            truncate_by_edge=True,
        )

    def to_area_of_interest(self) -> AreaOfInterest:
        """Convert the bounding box to a pyproj AreaOfInterest"""
        return AreaOfInterest(
            north_lat_degree=self.north,
            south_lat_degree=self.south,
            east_lon_degree=self.east,
            west_lon_degree=self.west,
        )

    def utm_crs_info(self) -> CRSInfo:
        """Find the CRSInfo for the containing UTM zone"""
        return query_utm_crs_info(
            datum_name=WGS84,
            area_of_interest=self.to_area_of_interest(),
            contains=True
        ).pop()

    def get_utm(self) -> CRS:
        """Get the CRS for the UTM zone that contains the bounding box"""
        crs_info = self.utm_crs_info()
        return CRS.from_authority(
            crs_info.auth_name,
            crs_info.code
        )
