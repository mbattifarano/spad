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
    assert not neighborhood.offset.isna().any()
    assert 'link_geometry' in neighborhood.columns
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


def test_map_match(osmnx_road_network, random_walk, gps_trajectory):
    links = ox.graph_to_gdfs(
        osmnx_road_network,
        nodes=False,
        edges=True,
    )
    spc = mm.ShortestPathCalculator(osmnx_road_network)
    path = mm.map_match(links, gps_trajectory, spc,
                        threshold=10.0, accuracy=1.0, scale=1.0)
    assert len(path) == len(random_walk)
    n_correct = 0
    for expected_link, idx in zip(random_walk, path):
        actual_link = mm.get_link_key(idx)
        n_correct += actual_link == expected_link
    # test that most links are correct
    assert n_correct / len(path) >= 0.9 or len(path) - n_correct <= 2
    # test that the true path is likely
    neighborhood = mm.link_neighborhood(links, gps_trajectory, threshold=10.0)
    neighborhood['p_emit'] = mm.emission_probabilities(neighborhood, 1.0)
    weight = mm.HMMEdgeWeight(neighborhood, spc, 1.0)
    path_cost = sum([weight(u, v, {}) for u, v in zip(path, path[1:])])
    _walk_idx = [(i,) + link for i, link in enumerate(random_walk)]
    walk_cost = sum([weight(u, v, {}) for u, v in zip(_walk_idx, _walk_idx[1:])])
    # path is minimal so walk cost should be at least as large
    assert path_cost <= walk_cost
    # but, the walk cost should be within 1% of the minimum
    assert walk_cost <= path_cost * 1.01

