"""Map Matching with a Hidden Markov Model

Contains an implementation of
Newson, Paul, and John Krumm.
"Hidden Markov map matching through noise and sparseness."
In Proceedings of the 17th ACM SIGSPATIAL international conference on advances
in geographic information systems, pp. 336-343. 2009.

See also:
https://github.com/valhalla/valhalla/blob/master/docs/meili/algorithms.md
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum, auto
import logging
import time
import datetime as dt
from typing import Union, Tuple, Iterable, List
import uuid
from tqdm import tqdm
import plyvel

import geopandas as gpd
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _dijkstra_multisource
import numpy as np
import pandas as pd
from toolz import curry
import osmnx as ox
from shapely.geometry import box, Polygon

from psycopg2.extras import register_uuid, execute_values

from .data import driver_segments, get_osmnx_network, get_osmnx_subnetwork
from .cache import LevelDBWriter

log = logging.getLogger(__name__)


class Terminals(Enum):
    """Virtual source and target nodes for the HMM directed graph"""

    source = auto()
    target = auto()


Node = Union[Terminals, pd.Series]
FloatOrColumnName = Union[str, float]
LinkKey = Tuple[int, int, int]

LINK_KEY_LEN = 3
EPS = np.finfo(np.float).tiny


def map_match_trajectories(
    conn,
    max_ping_time_delta: dt.timedelta,
    min_trajectory_duration: dt.timedelta,
    max_match_distance: float,
    transition_exp_scale: float = 1.0,
    subnetwork_buffer: float = 2000.0,
    lazy_load_network: bool = False,
    limit: int = None,
    commit_every: int = None,
    allow_reverse: bool = True,
    cache_path: str = None,
    start_at_driver: uuid.UUID = None,
):
    t0 = time.time()
    run_uuid = uuid.uuid4()
    log.info(f"Map-match run {run_uuid}.")
    if cache_path is None:
        cache_path = f"./{run_uuid}"
    db = plyvel.DB(cache_path, create_if_missing=True)
    if not lazy_load_network:
        log.info(f"Using entire OSM graph.")
        g, links = get_osmnx_network(conn)
        preserve_index_as_columns(links)
        links.sindex  # initialize the index
        spc = ShortestPathCalculator(g, allow_reverse=allow_reverse, lvldb=db)
        log.info(f"Built osmnx graph (t={time.time()-t0}s)")
    register_uuid()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO map_match_runs (
                run_id,
                created_at,
                max_ping_time_delta,
                min_trajectory_duration,
                max_match_distance,
                transition_exp_scale,
                subnetwork_buffer
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
            (
                run_uuid,
                dt.datetime.utcnow(),
                max_ping_time_delta,
                min_trajectory_duration,
                max_match_distance,
                transition_exp_scale,
                subnetwork_buffer,
            ),
        )
    n_complete = 0
    for segment in tqdm(
        driver_segments(
            conn, max_ping_time_delta, min_trajectory_duration, start_at_driver
        )
    ):
        segment_uuid = uuid.uuid4()
        initial_ping_id = segment.loc[0].index[0]
        log.info(
            f"Segment {segment_uuid} has {len(segment)} pings starting with "
            f"ping {initial_ping_id}"
        )
        if lazy_load_network:
            t0 = time.time()
            log.info(f"Building subnetwork graph.")
            g, links = get_osmnx_subnetwork(conn, segment, subnetwork_buffer)
            preserve_index_as_columns(links)
            links.sindex  # initialize the index
            spc = ShortestPathCalculator(
                g, allow_reverse=allow_reverse, lvldb=db
            )
            log.info(f"Built subnetwork in {time.time()-t0}s.")
        max_accuracy = segment.accuracy.max()
        if max_accuracy > max_match_distance:
            log.warning(
                f"Segment maximum accuracy exceeds maximum match distance: "
                f"{max_accuracy} > {max_match_distance} "
                f"(first ping id: {initial_ping_id})."
            )
        t0 = time.time()
        path = map_match(
            links,
            segment,
            spc,
            subnetwork_buffer=subnetwork_buffer,
            threshold=max_match_distance,
            scale=transition_exp_scale,
        )
        if path is None:
            continue
        log.info(
            f"Matched {len(path)} of {len(segment)} pings in "
            f"{time.time()-t0}s."
        )
        with conn.cursor() as cursor:
            values = [
                (run_uuid, segment_uuid, link_order) + tuple(values)
                for link_order, values in enumerate(
                    expand_path_links(spc, path)
                )
            ]
            n_items = len(values)
            execute_values(
                cursor,
                """INSERT INTO map_match (
                    run_id,
                    segment_id,
                    link_order,
                    ping_order,
                    ping_id,
                    u,
                    v,
                    key
                ) VALUES %s""",
                values,
            )
        log.info(f"Pending commit: {n_items} map matched links.")
        log.info(
            f"Shortest paths cache hit rate: {100*spc.stats.hit_rate:0.2f}%"
        )
        n_complete += 1
        if limit is not None and n_complete >= limit:
            log.info("Segment limit hit.")
            break
        if commit_every is not None and n_complete % commit_every == 0:
            log.info(f"Committing.")
            conn.commit()
    log.info("done.")


def preserve_index_as_columns(gdf: gpd.GeoDataFrame):
    """Add the index of gdf as column(s) in-place."""
    idx_df = gdf.index.to_frame()
    for col in idx_df.columns:
        gdf[col] = idx_df[col]


def create_map_match_tables(conn):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS map_match_runs (
                run_id uuid,
                created_at timestamp without time zone,
                max_ping_time_delta interval,
                min_trajectory_duration interval,
                max_match_distance real,
                transition_exp_scale real,
                subnetwork_buffer real
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS map_match (
                run_id uuid,
                segment_id uuid,
                ping_order integer,
                ping_id integer,
                link_order integer,
                u bigint,
                v bigint,
                key integer
            )
        """
        )


def map_match(
    links: gpd.GeoDataFrame,
    trajectory: gpd.GeoDataFrame,
    shortest_path_calculator: ShortestPathCalculator,
    subnetwork_buffer: float = 2000.0,
    threshold: float = 25.0,
    accuracy: FloatOrColumnName = "accuracy",
    scale: float = 1.0,
) -> List[tuple]:
    """Match a single gps trajectory to network links.

    The gps trajectory is projected into the coordinate reference system of the
    links GeoDataFrame which is assumed to have real distance units (e.g.
    meters)

    :param links: A GeoDataFrame of links in the network, should be of the form
        returned by osmnx.graph_to_gdfs
    :type links: gpd.GeoDataFrame
    :param trajectory: A single trajectory of temporally contiguous GPS pings
    :type trajectory: gpd.GeoDataFrame
    :param shortest_path_calculator: A ShortestPathCalculator for the
        underlying road network, which should be the same as the network that
        `links` belongs to.
    :type shortest_path_calculator: ShortestPathCalculator
    :param threshold: The maximum distance a gps ping can be from its matched
        link on the network, in units of the coordinate reference system of
        `links`.
    :type threshold: float
    :param accuracy: The name of the column in trajectory or a float to use as
        the "confidence radius" of each gps ping, that is the true location is
        within a circle of radius `accuracy` of the gps ping.
    :type accuracy: FloatOrColumnName
    :param scale: The scale parameter of an exponential distribution
        characterizing the transition probability.
    :type scale: float

    :return: A list of indices mapping each gps point to a link. Each element
        of the list is a tuple of the form (trajectory index | link index).
    :rtype: List[tuple]
    """
    t0 = time.time()
    trajectory = trajectory.to_crs(links.crs)
    log.info(
        f"Building trajectory link neighborhood (t={time.time()-t0:0.2f}s)."
    )
    neighborhood = link_neighborhood(
        links,
        trajectory,
        subnetwork_buffer=subnetwork_buffer,
        threshold=threshold,
    )
    if neighborhood is None:
        log.warning(
            "Link neighborhood creation failed. Skipping map matching."
        )
        return None
    log.info(f"Computing emission probabilities (t={time.time()-t0:0.2f}s).")
    neighborhood["p_emit"] = emission_probabilities(neighborhood, accuracy)
    log.info(f"Building HMM graph (t={time.time()-t0:0.2f}s).")
    hmm = build_hmm_graph(neighborhood, trajectory.index.names)
    log.info(
        f"HMM has {hmm.number_of_nodes()} nodes and {hmm.number_of_edges()} "
        f"edges (t={time.time()-t0:0.2f}s)."
    )
    log.info(f"Computing HMM edge weights (t={time.time()-t0:0.2f}s).")
    weight = HMMEdgeWeight(neighborhood, shortest_path_calculator, scale)
    log.info(f"Solving the HMM (t={time.time()-t0:0.2f}s).")
    path = nx.shortest_path(hmm, Terminals.source, Terminals.target, weight)
    log.info(f"finished map matching in {time.time()-t0:0.2f}s.")
    return path[1:-1]


def plot_map_match(g, trajectory, path, **plot_kws):
    kws = dict(
        orig_dest_size=400,
        figsize=(12, 20),
        show=False,
        close=False,
    )
    kws.update(plot_kws)
    nodes = path_to_nodes(g, path)
    fig, ax = ox.plot_graph_route(g, nodes, **kws)
    pings = trajectory.to_crs(g.graph["crs"])
    pings.plot(ax=ax, zorder=1, markersize=200, color="blue")
    for i, (_, row) in enumerate(pings.iterrows()):
        x = row.geometry.x
        y = row.geometry.y
        ax.annotate(str(i), xy=(x, y), color="w", fontsize=32)
    for i, u in enumerate(nodes):
        x = g.nodes[u]["x"]
        y = g.nodes[u]["y"]
        ax.annotate(str(i), xy=(x, y), color="r", fontsize=32)


def path_to_nodes(spc: ShortestPathCalculator, path):
    nodes = []
    for idx in path:
        u, v, _ = get_link_key(idx)
        if nodes:
            # if there are nodes in the list so far
            if (u, v) == tuple(nodes[-2:]):
                # ignore repeat edges
                continue
            else:
                v0 = nodes.pop()
                if spc.g.number_of_edges(v0, u):
                    # if there is an edge from v0 to u add it
                    nodes.append(v0)
                    nodes.append(u)
                else:
                    # else add the shortest path
                    nodes.extend(spc.get_path(v0, u))
                nodes.append(v)
        else:
            # just add the link
            nodes.append(u)
            nodes.append(v)
    return nodes


def expand_path_links(spc: ShortestPathCalculator, path):
    prev_link = None
    for idx in path:
        link = u, v, k = get_link_key(idx)
        if prev_link is None:
            # first link
            yield idx
        else:
            _, v_prev, _ = prev_link
            if link == prev_link:
                # consecutive gps pings assigned to same link
                yield idx
            elif v_prev == u:
                # Edges are directly connected
                yield idx
            else:
                # Add edges along the shortest path from v_prev to u
                for link_idx in _links_on_shortest_path(spc, v_prev, u):
                    yield with_null_gps_key(len(idx), link_idx)
                yield idx
        prev_link = (u, v, k)


def _links_on_shortest_path(spc: ShortestPathCalculator, s, t):
    nodes = spc.get_path(s, t)
    for u, v in zip(nodes, nodes[1:]):
        _, k = min((data["length"], key) for key, data in spc.g[u][v].items())
        yield u, v, k


def with_null_gps_key(idx_len: int, link_key: tuple) -> tuple:
    null_gps_key = (None,) * (idx_len - LINK_KEY_LEN)
    return null_gps_key + link_key


def build_hmm_graph(
    neighborhood: gpd.GeoDataFrame, gps_keys: Iterable[str]
) -> nx.DiGraph:
    """Construct a directed graph representing the possible HHM paths.

    :param neighborhood: The link neighborhood for each gps ping
    :type neighborhood: gpd.GeoDataFrame
    :param gps_keys: The index names of the gps trajectory GeoDataFrame
    :type gps_keys: Iterable[str]

    :return: The HMM as a directed graph
    :rtype: nx.DiGraph
    """
    hmm = nx.DiGraph()
    ping_neighborhoods = neighborhood.groupby(list(gps_keys))
    predecessor_candidates = [Terminals.source]
    for gps_key, candidates in ping_neighborhoods:
        for u in predecessor_candidates:
            for v in candidates.index:
                hmm.add_edge(u, v)
        predecessor_candidates = candidates.index
    for u in predecessor_candidates:
        hmm.add_edge(u, Terminals.target)
    return hmm


class HMMEdgeWeight:
    """Edge weight function for the map-matching HMM graph"""

    def __init__(
        self,
        neighborhood: gpd.GeoDataFrame,
        spc: ShortestPathCalculator,
        scale: float,
    ):
        """Create an edge weight function to use with an HMM directed graph

        Instances of this class may be directly used as edge weight functions
        expected by networkx shortest path functions.

        :param neighborhood: The link neighborhood for each gps ping
        :type neighborhood: gpd.GeoDataFrame
        :param spc: The shortest path calculator for the underlying network
        :type spc: ShortestPathCalculator
        :param scale: The scale parameter of an exponential distribution
        characterizing the transition probability.
        """
        self.neighborhood = neighborhood
        self.spc = spc
        self.scale = scale

    def __call__(self, u, v, data) -> float:
        u_row = self._get_node_data(u)
        v_row = self._get_node_data(v)
        return node_cost(v_row) + edge_cost(
            self.spc, self.neighborhood.geometry.name, u_row, v_row, self.scale
        )

    def _get_node_data(self, u) -> Node:
        return u if is_terminal(u) else self.neighborhood.loc[u]


def is_terminal(u) -> bool:
    """Return True if u is an member of the Terminals enum"""
    return isinstance(u, Terminals)


def edge_cost(
    shortest_path_calculator: ShortestPathCalculator,
    pt_geom_column: str,
    u: Node,
    v: Node,
    scale: float = 1.0,
):
    """Compute the weight of an edge in the map-matching HMM graph"""
    if is_terminal(u) or is_terminal(v):
        return 0.0
    return -np.log(
        transition_probabilities(
            shortest_path_calculator, pt_geom_column, u, v, scale
        )
        + EPS
    )


def node_cost(u: Node):
    """Compute the weight of a node in the map-matching HMM graph

    :param u: A node of the HMM graph
    :type u: Node
    :return: The node cost
    :rtype: float
    """
    return 0.0 if is_terminal(u) else -np.log(u.p_emit + EPS)


def safe_neg_log(value: float) -> float:
    """Returns the negative log of value plus a small constant

    Value must be non-negative. Zero values will return a large positive number

    :param value: A non-negative number
    :type value: float
    :return: -log(value)
    :rtype: float
    """
    return -np.log(value + EPS)


def get_link_key(compound_index: tuple) -> LinkKey:
    """Retrieve the link index from a compound index.

    It is assumed that the link key forms the tail of the compound index.
    """
    return tuple(map(int, compound_index[-LINK_KEY_LEN:]))


def get_gps_key(compound_index: tuple) -> Tuple:
    return tuple(map(int, compound_index[:-LINK_KEY_LEN]))


def link_neighborhood(
    links: gpd.GeoDataFrame,
    trajectory: gpd.GeoDataFrame,
    subnetwork_buffer: float,
    threshold: float,
) -> gpd.GeoDataFrame:
    """Find the link neighborhood of each point in the trajectory.

    This function assumes that both geo dataframes are in the same projected
    coordinate reference system in units of meters.

    :param links: A GeoDataFrame of links in a road network (e.g. the output of
        osmnx.graph_to_gdfs)
    :type links: gpd.GeoDataFrame
    :param trajectory: A GeoDataFrame of points
    :type trajectory: gpd.GeoDataFrame
    :param threshold: The maximum distance (in projection units) between a
        point and the links in its neighborhood
    :type threshold: float

    :return: The link neighborhood of each point in trajectory. Retains all
        columns from links and trajectory and is indexed by a compound index of
        the form (trajectory.index | links.index)
    :rtype: gpd.GeoDataFrame
    """
    links["link_geometry"] = links.geometry  # keep the geom around
    subnetwork_idx = nearby_link_idx(links, trajectory, subnetwork_buffer)
    subnet_links = links.iloc[subnetwork_idx]
    while True:
        points = trajectory.set_geometry(
            buffered_geometry(trajectory, threshold)
        )
        neighborhood = gpd.sjoin(
            points,
            subnet_links,
            how="left",
            predicate="intersects",
        )
        failed_pings = neighborhood.osmid.isna()
        if failed_pings.any():
            failed_ping_ids = list(
                neighborhood[failed_pings].index.get_level_values("id")
            )
            log.warning(
                f"{failed_pings.sum()} gps pings failed to match a link "
                f"within {threshold} meters; re-trying with double the "
                f"threshold. failed pings: {failed_ping_ids}."
            )
            if threshold > subnetwork_buffer:
                log.warning(
                    "The match distance threshold is larger than the "
                    "subnetwork buffer. This likely means the pings "
                    "are out of range of the network. Refusing to map "
                    "match this segment."
                )
                return None
            threshold = threshold * 2
        else:
            break
    neighborhood.set_geometry(trajectory.geometry, inplace=True)
    neighborhood.set_index(list(links.index.names), inplace=True, append=True)
    neighborhood["distance_to_link"] = neighborhood.distance(
        neighborhood.link_geometry
    )
    neighborhood["offset"] = neighborhood.apply(
        _offset_on_link(neighborhood.geometry.name), axis=1
    )
    return neighborhood


BBOX_EXPAND = np.array([-1.0, -1.0, 1.0, 1.0])


def buffered_bbox(gdf: gpd.GeoDataFrame, buffer: float) -> Polygon:
    return box(*(gdf.total_bounds + buffer * BBOX_EXPAND))


def nearby_link_idx(
    links: gpd.GeoDataFrame, trajectory: gpd.GeoDataFrame, buffer: float
) -> np.array:
    return links.sindex.query(buffered_bbox(trajectory, buffer), sort=True)


@curry
def _offset_on_link(pt_geom_column: str, row: pd.Series) -> float:
    try:
        return row.link_geometry.project(row[pt_geom_column])
    except AttributeError:
        log.warning(
            f"Neighborhood row is missing a geometry; returning nan "
            f"(row: {row})"
        )
        return np.nan


def emission_probabilities(
    neighborhood: gpd.GeoDataFrame, accuracy: FloatOrColumnName
) -> gpd.GeoSeries:
    """Compute emission probabilities for each link the trajectory neighborhood

    :param neighborhood: A GeoDataFrame containing the link neighborhood for
        each point in the gps trajectory, as returned by ``link_neighborhood``
    :type neighborhood: gpd.GeoDataFrame
    :param accuracy: The accuracy of the trajectory, used as the standard
        deviation. Can be a float, the name of a column in neighborhood, or a
        Series
    :type accuracy: Union[float, str, pd.Series]

    :return: The emission probability of each candidate link
    :rtype: gpd.GeoSeries
    """
    if isinstance(accuracy, str):
        accuracy = neighborhood[accuracy]
    z = accuracy * np.sqrt(2 * np.pi)
    p = np.exp(-0.5 * (neighborhood.distance_to_link / accuracy) ** 2)
    return p / z


def transition_probabilities(
    shortest_path_calculator: ShortestPathCalculator,
    pt_geom_column: str,
    u_row: pd.Series,
    v_row: pd.Series,
    scale: float = 1.0,
) -> float:
    """Compute transition probabilities between successive point-link pairs.

    :param shortest_path_calculator: An instance of ShortestPathCalculator for
        the underlying road network
    :type shortest_path_calculator: ShortestPathCalculator
    :param pt_geom_column: The name of the geometry column of neighborhood
    :type pt_geom_column: str
    :param u_row: A row from neighborhood
    :type u_row: pd.Series
    :param v_row: A row from the neighborhood of the next gps ping
    :type v_row: pd.Series
    :param scale: The scale parameter of an exponential distribution
    :type scale: float

    :return: The transition probability
    :rtype: float
    """
    routable_distance = shortest_path_calculator.distance(u_row, v_row)
    measurement_distance = u_row[pt_geom_column].distance(
        v_row[pt_geom_column]
    )
    d = np.abs(routable_distance - measurement_distance)
    return scale * np.exp(-scale * d)


def buffered_geometry(
    points: gpd.GeoDataFrame, radius: float
) -> gpd.GeoSeries:
    """Compute a square buffer around each point in points

    The buffers are computed as the circumscribing square of a circular buffer
    of a given radius from each point. That is, a square centered at each point
    with side length 2 * radius.

    :param points: A GeoDataFrame of point geometries
    :type points: gpd.GeoDataFrame
    :param radius: buffer radius, in units of the projection
    :type radius: float
    """
    return points.buffer(
        distance=radius,
        resolution=1,
    ).envelope


NULL_NODE = -1


class ShortestPathCalculator:
    """Compute shortest paths on a graph accounting for position on a link"""

    def __init__(
        self,
        g: nx.MultiDiGraph,
        allow_reverse: bool = True,
        lvldb: plyvel.DB = None,
    ):
        """
        :param g: A routable network in the form returned by osmnx
        :type g: nx.MultiDiGraph
        :param allow_reverse: Relevant when start and end locations are on the
            same link, but the end location is behind the start direction with
            respect to the direction of travel. If True, a vehicle is allowed
            to reverse on the link, in which case the distance will be
            negative, if False, the vehicle must "circle the block".
        :type allow_reverse: bool
        """
        self.g = g
        self.dist_pred_cache = LevelDBWriter(lvldb, "ll", "dl")
        self.cache = defaultdict(dict)
        self.predecessors = defaultdict(dict)
        self.stats = CacheStats()
        self.allow_reverse = allow_reverse

    def get_path(self, u, v):
        if self.dist_pred_cache.get((u, v), None) is None:
            self.stats.miss()
            self.shortest_path(u, v)
        else:
            self.stats.hit()
        nodes = [v]
        current = v
        while current != u:
            try:
                _, current = self.dist_pred_cache.get((u, current))
            except KeyError:
                log.error(
                    f"Failed to find predecessor for {current} on path from "
                    f"{u}->{v}, returning partial path."
                )
                nodes.reverse()
                return nodes
            if current == NULL_NODE:
                log.error(
                    f"Failed to find predecessor for {current} on path from "
                    f"{u}->{v}, returning partial path."
                )
                nodes.reverse()
                return nodes
            nodes.append(current)
        nodes.reverse()
        return nodes

    def distance(self, s_row: pd.Series, t_row: pd.Series) -> float:
        """Compute the shortest routable distance between nodes s and t

        :param s_row: The row of neighborhood to start from
        :type s_row: pd.Series
        :param t_row: The row of neighborhood to end at
        :type t_row: pd.Series
        :return: The distance travelled on the network from s to t
        :rtype: float

        Terminology:

        projection
            The projection of the gps point onto the link is the point on the
            link closest to the gps ping.

        offset
            The offset is the distance from the start of the link to the
            projection of the gps point onto the link

        The network distance is then the sum of the following:

        - the distance from the projection of s_pt to the end of the link
        - the shortest path distance from the end node of s_link to the start
          node of t_link
        - the distance from the start node of t_link to the projection of t_pt
          onto t_link

        EXCEPT if ``s_link == t_link``.
        This occurs when the two pings are on the same link. In this case
        return ``t_offset - s_offset``. This will be negative if the motion is
        against the flow of traffic and ``allow_reverse`` is True. In the
        context of the transition probabilities this will be helpful to make
        such motion less likely, but still more likely than circling the block
        to get to an earlier point on the link, which will happen if
        ``allow_reverse`` is False.
        """
        s_offset = s_row.offset
        t_offset = t_row.offset
        s_link = get_link_key(s_row.name)
        t_link = get_link_key(t_row.name)
        if s_link == t_link:
            d = t_offset - s_offset
            if d >= 0 or self.allow_reverse:
                return d
        s_length = s_row.link_geometry.length
        _, s, _ = s_link
        t, _, _ = t_link
        try:
            dist, _ = self.dist_pred_cache.get((s, t))
            self.stats.hit()
        except KeyError:
            dist = self.shortest_path(s, t)
            self.stats.miss()
        return (
            (s_length - s_offset)  # distance from s pt to end of s link
            + dist  # path length from s->t
            + t_offset  # distance from beginning of t link to t pt
        )

    def shortest_path(self, source, target=None) -> dict:
        pred = {source: []}
        dist = _dijkstra_multisource(
            self.g,
            [source],
            target=target,
            weight=self._link_length,
            pred=pred,
        )
        if target is not None and target not in dist:
            dist[target] = np.inf
            log.warning(f"No path found from {source} to {target}.")
        with self.dist_pred_cache.db.write_batch() as batch:
            for v, distance in dist.items():
                predecessor = (
                    NULL_NODE
                    if (np.isinf(distance) or v == source)
                    else pred[v][0]
                )
                self.dist_pred_cache.put(
                    (source, v), (distance, predecessor), batch=batch
                )
        return dist[target]

    @staticmethod
    def _link_length(u, v, data):
        return min(attr["length"] for attr in data.values())


class CacheStats:
    """Record cache hits and misses and report basic statistics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0

    @property
    def calls(self) -> int:
        """Number of hits and misses"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of calls that were hits"""
        return self.hits / self.calls

    @property
    def miss_rate(self) -> float:
        """Fraction of calls that were misses"""
        return self.misses / self.calls

    def hit(self):
        """Record a hit"""
        self.hits += 1

    def miss(self):
        """Record a miss"""
        self.misses += 1
