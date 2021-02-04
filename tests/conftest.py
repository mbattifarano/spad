import os
from pytest import fixture
import osmnx as ox
import numpy as np
import geopandas as gpd


@fixture(scope="session")
def fixtures_directory():
    return os.path.join("tests", "fixtures")


@fixture(scope="session")
def mock_gps_data(fixtures_directory):
    return os.path.join(
        fixtures_directory,
        "mock_gps.csv"
    )


@fixture(scope="session")
def rng():
    """Random number generator with a fixed seed"""
    return np.random.default_rng(0)


@fixture(scope='session')
def pittsburgh_utm():
    """The EPSG code for the UTM containing Pittsburgh (and Ohio)"""
    return 'EPSG:32617'


@fixture
def pittsburgh_bbox_lat_lons():
    """A convenient bounding box roughly centered on pittsburgh."""
    return (
        -80.5,
        40.0,
        -79.5,
        41.0
    )


@fixture
def point_lat_lon():
    """The lat lon location of Porter Hall on Carnegie Mellon's campus"""
    return 40.44162943711366, -79.94628773614849


@fixture
def osmnx_road_network(point_lat_lon, pittsburgh_utm):
    """A small road network around Carnegie Mellon's campus
    
    Convert to geo data frame and then back to a graph to ensure all edges have
    a geometry attribute.
    https://stackoverflow.com/questions/64333794/osmnx-graph-from-point-and-geometry-information/64376567#64376567
    """
    g = ox.project_graph(
        ox.graph_from_point(
            point_lat_lon,
            dist=750,
            network_type='drive',
            truncate_by_edge=True,
        ),
        pittsburgh_utm
    )
    nodes, edges = ox.graph_to_gdfs(g, fill_edge_geometry=True)
    return ox.graph_from_gdfs(nodes, edges, graph_attrs=g.graph)


@fixture
def random_walk(rng, osmnx_road_network):
    """Return a random walk on a road network as a list of edges"""
    i = 0
    n = 25
    u = rng.choice(list(osmnx_road_network.nodes))
    neighbors = list(osmnx_road_network.neighbors(u))
    walk = []
    while i < n and neighbors:
        i += 1
        v = rng.choice(neighbors)
        walk.append((u, v, 0))
        u = v
        neighbors = list(osmnx_road_network.neighbors(u))
    return walk


@fixture
def gps_trajectory(rng, osmnx_road_network, random_walk):
    """Generate a noisy gps trajectory from a random walk"""
    sigma = 0.5
    noise = rng.normal(0, sigma, size=(len(random_walk), 2))
    trajectory = []
    ids = []
    sigmas = []
    for i, (u, v, k) in enumerate(random_walk):
        x, y = rng.choice(
            osmnx_road_network.get_edge_data(u, v, k)['geometry'].coords
        )
        eps_x, eps_y = noise[i]
        ids.append(i)
        sigmas.append(sigma)
        trajectory.append((
            x + eps_x,
            y + eps_y
        ))
    xs, ys = zip(*trajectory)
    return gpd.GeoDataFrame(dict(
        id=ids,
        accuracy=sigmas,
        locations=gpd.points_from_xy(xs, ys,
                                     crs=osmnx_road_network.graph['crs'])
    )).set_geometry('locations').set_index('id')
