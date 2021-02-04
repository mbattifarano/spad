import geopandas as gpd
import osmnx as ox
import pytest
from spad import map_matching as mm


def test_buffered_geometry():
    xy = [
        (0, 0),
        (0, 5),
        (5, 0)
    ]
    points = gpd.GeoDataFrame(
        {'locations': gpd.points_from_xy(*zip(*xy))}
    ).set_geometry('locations')
    radius = 2.0
    buffers = mm.buffered_geometry(points, radius)
    assert buffers.contains(points).all()
    assert buffers.centroid.geom_almost_equals(points).all()
    assert (buffers.area == (2 * radius) ** 2).all()


def test_link_neighborhood(osmnx_road_network, gps_trajectory):
    links = ox.graph_to_gdfs(
        osmnx_road_network,
        nodes=False,
        edges=True,
    )
    assert links.crs == gps_trajectory.crs, \
        "Fixture CRS should match, check conftest.py"
    radius = 200
    # the link neighborhood is a square of side length radius centered at
    # each point. The maximum distance is half the diagonal of the square:
    max_distance = 0.5 * (2 * (2 * radius)**2)**0.5
    neighborhood = mm.link_neighborhood(links, gps_trajectory, radius)
    assert links.crs == neighborhood.crs
    assert (len(neighborhood.groupby(gps_trajectory.index.names))
            == len(gps_trajectory))
    assert not neighborhood.distance_to_link.isna().any()
    for (id_, u, v, key), row in neighborhood.iterrows():
        loc = row.locations
        pt = gps_trajectory.geometry.loc[id_]
        link = links.geometry.loc[(u, v, key)]
        assert loc.equals(pt)
        assert loc.distance(link) == pytest.approx(row.distance_to_link)
        assert pt.distance(link) == pytest.approx(row.distance_to_link)
        assert row.distance_to_link >= 0
        assert row.distance_to_link <= max_distance
    p1 = mm.emission_probabilities(neighborhood, 1.0)
    assert p1.min() >= 0.0
    assert p1.max() <= 1.0
    p2 = mm.emission_probabilities(neighborhood, 'accuracy')
    assert p2.min() >= 0.0
    assert p2.max() <= 1.0
    assert p2.index.names == ('id', 'u', 'v', 'key')
    p3 = mm.emission_probabilities(neighborhood, neighborhood.accuracy)
    assert p3.min() >= 0.0
    assert p3.max() <= 1.0
    assert (p2 == p3).all()
