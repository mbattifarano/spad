"""pytest fixtures."""
import os
from pytest import fixture
import osmnx as ox
import numpy as np
import geopandas as gpd
from scipy.sparse import coo_matrix


@fixture(scope="session")
def fixtures_directory():
    """Return the fixtures directory, relative to project root."""
    return os.path.join("tests", "fixtures")


@fixture(scope="session")
def mock_gps_data(fixtures_directory):
    """Path to the mock GPS data."""
    return os.path.join(fixtures_directory, "mock_gps.csv")


@fixture(scope="session")
def rng():
    """Return a andom number generator with a fixed seed."""
    return np.random.default_rng(0)


@fixture(scope="session")
def pittsburgh_utm():
    """Return the EPSG code for the UTM containing Pittsburgh (and Ohio)."""
    return "EPSG:32617"


@fixture
def pittsburgh_bbox_lat_lons():
    """Return a convenient bounding box roughly centered on pittsburgh."""
    return (-80.5, 40.0, -79.5, 41.0)


@fixture
def point_lat_lon():
    """Return the lat lon location of Porter Hall on CMU campus."""
    return 40.44162943711366, -79.94628773614849


@fixture
def osmnx_road_network(point_lat_lon, pittsburgh_utm):
    """Return a small road network around Carnegie Mellon's campus.

    Convert to geo data frame and then back to a graph to ensure all edges have
    a geometry attribute.
    https://stackoverflow.com/questions/64333794/osmnx-graph-from-point-and-geometry-information/64376567#64376567
    """
    g = ox.project_graph(
        ox.graph_from_point(
            point_lat_lon,
            dist=750,
            network_type="drive",
            truncate_by_edge=True,
        ),
        pittsburgh_utm,
    )
    nodes, edges = ox.graph_to_gdfs(g, fill_edge_geometry=True)
    return ox.graph_from_gdfs(nodes, edges, graph_attrs=g.graph)


@fixture
def random_walk(rng, osmnx_road_network):
    """Return a random walk on a road network as a list of edges."""
    i = 0
    n = 25
    u = rng.choice(list(osmnx_road_network.nodes))
    neighbors = list(osmnx_road_network.neighbors(u))
    walk = []
    while i < n and neighbors:
        v = rng.choice(neighbors)
        if walk and walk[-1] == (v, u, 0):
            # don't allow a u->v->u walk, remove v from neighbors
            neighbors.remove(v)
        else:
            i += 1
            walk.append((u, v, 0))
            neighbors = list(osmnx_road_network.neighbors(v))
            u = v
    return walk


@fixture
def gps_trajectory(rng, osmnx_road_network, random_walk):
    """Generate a noisy gps trajectory from a random walk."""
    sigma = 0.5
    noise = rng.normal(0, sigma, size=(len(random_walk), 2))
    trajectory = []
    ids = []
    sigmas = []
    for i, (u, v, k) in enumerate(random_walk):
        coords = osmnx_road_network.get_edge_data(u, v, k)["geometry"].coords
        x, y = rng.choice(coords)
        eps_x, eps_y = noise[i]
        ids.append(i)
        sigmas.append(sigma)
        trajectory.append((x + eps_x, y + eps_y))
    xs, ys = zip(*trajectory)
    return (
        gpd.GeoDataFrame(
            dict(
                id=ids,
                accuracy=sigmas,
                locations=gpd.points_from_xy(
                    xs, ys, crs=osmnx_road_network.graph["crs"]
                ),
            )
        )
        .set_geometry("locations")
        .set_index("id")
    )


@fixture
def braess_links():
    """Return the node pairs of the Braess network links."""
    return [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]


@fixture
def braess_ue_link_flow(braess_links):
    """Return the UE Braess link flow as a sparse matrix."""
    n_nodes = 4
    row, col = zip(*braess_links)
    flow = np.array([4.0, 2.0, 2.0, 2.0, 4.0])
    return coo_matrix((flow, (row, col)), shape=(n_nodes, n_nodes))


@fixture
def braess_od_demand():
    """Return the Braess network OD demand matrix."""
    n_nodes = 4
    return coo_matrix(
        ([6.], ([0], [3])),
        shape=(n_nodes, n_nodes)
    )


@fixture
def braess_link_cost():
    """Return the Braess network link cost function."""
    def lpf(x):
        m = np.array([10., 1., 1., 1., 10.])
        b = np.array([0., 50., 10., 50., 0.])
        return np.multiply(m, x) + b
    return lpf


@fixture
def braess_ue_link_cost(braess_link_cost, braess_ue_link_flow):
    """Return the Braess link cost at UE, as a sparse matrix."""
    data = braess_link_cost(braess_ue_link_flow.data)
    row = braess_ue_link_flow.row
    col = braess_ue_link_flow.col
    shape = braess_ue_link_flow.shape
    return coo_matrix(
        (data, (row, col)),
        shape
    )
