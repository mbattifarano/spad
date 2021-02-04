import pytest
from pyproj import Geod

from spad.bounding_box import BoundingBox
from spad.etl import read_csv, Columns


def test_bounding_box():
    bbox = BoundingBox(
        min_lat=0,
        min_lon=1,
        max_lat=2,
        max_lon=3
    )
    assert bbox.north == 2
    assert bbox.south == 0
    assert bbox.east == 3
    assert bbox.west == 1


def test_bounding_box_from_geodataframe(mock_gps_data):
    df = read_csv(mock_gps_data)
    bbox = BoundingBox.from_geodataframe(df)
    lons = df[Columns.lon.name]
    lats = df[Columns.lat.name]
    assert ((bbox.west <= lons) & (lons <= bbox.east)).all()
    assert ((bbox.south <= lats) & (lats <= bbox.north)).all()


def test_bounding_box_utm_info(pittsburgh_bbox_lat_lons, pittsburgh_utm):
    bbox = BoundingBox(*pittsburgh_bbox_lat_lons)
    crsinfo = bbox.utm_crs_info()
    auth_name, code = pittsburgh_utm.split(':')
    assert crsinfo.auth_name == auth_name
    assert crsinfo.code == code
    assert 'UTM zone 17N' in crsinfo.name


def test_bounding_box_utm_distances(mock_gps_data, pittsburgh_utm):
    df = read_csv(mock_gps_data)
    lons = df[Columns.lon.name].values
    lats = df[Columns.lat.name].values
    bbox = BoundingBox.from_geodataframe(df)
    geod = Geod(ellps='WGS84')
    # calculate the distances (in meters) between the first and second points
    _, _, distance = geod.inv(
        lons1=lons[0],
        lats1=lats[0],
        lons2=lons[1],
        lats2=lats[1],
    )
    crs = bbox.get_utm()
    assert crs.srs == pittsburgh_utm
    utm_df = df.to_crs(crs)
    utm_pt = utm_df.iloc[0].geometry
    # UTM is in meters so the distances should match
    distances = utm_df.distance(utm_pt).values
    assert distances[0] == 0
    assert distances[1] == pytest.approx(distance, abs=0.5)
