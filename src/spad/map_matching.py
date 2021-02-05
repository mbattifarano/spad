"""Map Matching with a Hidden Markov Model

Implementation of:

Hidden Markov map matching through noise and sparseness
Newson, Paul, and John Krumm
2009

"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum, auto
from heapq import heappush, heappop
from itertools import count
from typing import Union, Tuple, Iterable, List

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from toolz import curry


class Terminals(Enum):
    """Virtual source and target nodes for the HMM directed graph"""
    source = auto()
    target = auto()


Node = Union[Terminals, pd.Series]
FloatOrColumnName = Union[str, float]
LinkKey = Tuple[int, int, int]

LINK_KEY_LEN = 3
EPS = np.finfo(np.float).tiny


def map_match(links: gpd.GeoDataFrame, trajectory: gpd.GeoDataFrame,
              shortest_path_calculator: ShortestPathCalculator,
              threshold: float = 200.0,
              accuracy: FloatOrColumnName = 'accuracy',
              scale: float = 1.0) -> List[tuple]:
    """Match a single gps trajectory to network links.

    The gps trajectory is projected into the coordinate reference system of the
    links GeoDataFrame which is assumed to have real distance units (e.g.
    meters)

    :param links: A GeoDataFrame of links in the network, should be of the form
    returned by osmnx.graph_to_gdfs
    :type links: gpd.GeoDataFrame
    :param trajectory: A single trajectory of temporally contiguous GPS pings
    :type trajectory: gpd.GeoDataFrame
    :param shortest_path_calculator: A ShortestPathCalculator for the underlying
    road network, which should be the same as the network that `links` belongs
    to.
    :type shortest_path_calculator: ShortestPathCalculator
    :param threshold: The maximum distance a gps ping can be from its matched
    link on the network, in units of the coordinate reference system of `links`
    :type threshold: float
    :param accuracy: The name of the column in trajectory or a float to use as
    the "confidence radius" of each gps ping, that is the true location is
    within a circle of radius `accuracy` of the gps ping.
    :type accuracy: FloatOrColumnName
    :param scale: The scale parameter of an exponential distribution
    characterizing the transition probability.
    :type scale: float

    :return: A list of indices mapping each gps point to a link. Each element of
    the list is a tuple of the form (trajectory index | link index).
    :rtype: List[tuple]
    """
    trajectory = trajectory.to_crs(links.crs)
    neighborhood = link_neighborhood(links, trajectory, threshold)
    neighborhood['p_emit'] = emission_probabilities(neighborhood, accuracy)
    hmm = build_hmm_graph(neighborhood, trajectory.index.names)
    weight = HMMEdgeWeight(neighborhood, shortest_path_calculator, scale)
    path = nx.shortest_path(hmm, Terminals.source, Terminals.target, weight)
    return path[1:-1]


def build_hmm_graph(neighborhood: gpd.GeoDataFrame,
                    gps_keys: Iterable[str]) -> nx.DiGraph:
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
    def __init__(self, neighborhood: gpd.GeoDataFrame,
                 spc: ShortestPathCalculator, scale: float):
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
        return (
                node_cost(v_row)
                + edge_cost(self.spc, self.neighborhood.geometry.name,
                            u_row, v_row, self.scale)
        )

    def _get_node_data(self, u) -> Node:
        return u if is_terminal(u) else self.neighborhood.loc[u]


def is_terminal(u) -> bool:
    """Return True if u is an member of the Terminals enum"""
    return isinstance(u, Terminals)


def edge_cost(shortest_path_calculator: ShortestPathCalculator,
              pt_geom_column: str,
              u: Node, v: Node, scale: float = 1.0):
    """Compute the weight of an edge in the map-matching HMM graph"""
    if is_terminal(u) or is_terminal(v):
        return 0.0
    return -np.log(
        transition_probabilities(shortest_path_calculator, pt_geom_column,
                                 u, v, scale)
        + EPS
    )


def node_cost(u: Node):
    """Compute the weight of a node in the map-matching HMM graph"""
    return 0.0 if is_terminal(u) else -np.log(u.p_emit + EPS)


def safe_neg_log(value: float) -> float:
    """Returns the negative log of value plus a small constant

    Value must be non-negative. Zero values will return a large positive number
    """
    return -np.log(value + EPS)


def get_link_key(compound_index: tuple) -> LinkKey:
    """Retrieve the link index from a compound index.

    It is assumed that the link key forms the tail of the compound index.
    """
    return compound_index[-LINK_KEY_LEN:]


def _get_gps_key(compound_index: tuple) -> Tuple:
    return compound_index[:-LINK_KEY_LEN]


def link_neighborhood(links: gpd.GeoDataFrame, trajectory: gpd.GeoDataFrame,
                      threshold: float = 250) -> gpd.GeoDataFrame:
    """Find the link neighborhood of each point in the trajectory.

    This function assumes that both geo dataframes are in the same projected
    coordinate reference system in units of meters.

    :param links: A GeoDataFrame of links in a road network (e.g. the output of
    osmnx.graph_to_gdfs)
    :type links: gpd.GeoDataFrame
    :param trajectory: A GeoDataFrame of points
    :type trajectory: gpd.GeoDataFrame
    :param threshold: The maximum distance (in projection units) between a point
    and the links in its neighborhood
    :type threshold: float
    """
    points = trajectory.set_geometry(buffered_geometry(trajectory, threshold))
    links['link_geometry'] = links.geometry  # keep the geom around
    neighborhood = gpd.sjoin(
        points,
        links.reset_index(),  # sjoin removes the index names, keep them.
        how="left",
        op="intersects",
    )
    neighborhood.set_geometry(trajectory.geometry, inplace=True)
    neighborhood.set_index(list(links.index.names),
                           inplace=True, append=True)
    neighborhood['distance_to_link'] = neighborhood.distance(
        neighborhood.link_geometry
    )
    neighborhood['offset'] = neighborhood.apply(
        _offset_on_link(neighborhood.geometry.name),
        axis=1
    )
    return neighborhood


@curry
def _offset_on_link(pt_geom_column: str, row: pd.Series) -> float:
    return row.link_geometry.project(row[pt_geom_column])


def emission_probabilities(neighborhood: gpd.GeoDataFrame,
                           accuracy: FloatOrColumnName) -> gpd.GeoSeries:
    """Compute emission probabilities for each link the trajectory neighborhood

    :param neighborhood: A GeoDataFrame containing the link neighborhood for
    each point in the gps trajectory, as returned by ``link_neighborhood``
    :type neighborhood: gpd.GeoDataFrame
    :param accuracy: The accuracy of the trajectory, used as the standard
    deviation. Can be a float, the name of a column in neighborhood, or a Series
    :type accuracy: Union[float, str, pd.Series]
    """
    if isinstance(accuracy, str):
        accuracy = neighborhood[accuracy]
    z = accuracy * np.sqrt(2 * np.pi)
    p = np.exp(-0.5 * (neighborhood.distance_to_link / accuracy) ** 2)
    return p / z


def transition_probabilities(shortest_path_calculator: ShortestPathCalculator,
                             pt_geom_column: str,
                             u_row: pd.Series, v_row: pd.Series,
                             scale: float = 1.0) -> float:
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
    measurement_distance = u_row[pt_geom_column].distance(v_row[pt_geom_column])
    d = np.abs(routable_distance - measurement_distance)
    return scale * np.exp(-scale * d)


def buffered_geometry(points: gpd.GeoDataFrame, radius: float) -> gpd.GeoSeries:
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


class ShortestPathCalculator:
    """Compute shortest paths on a graph accounting for position within a link
    """
    def __init__(self, g: nx.MultiDiGraph, allow_reverse: bool = True):
        """Compute shortest distance paths on a graph

        :param g: A routable network in the form returned by osmnx
        :type g: nx.MultiDiGraph
        :param allow_reverse: Relevant when start and end locations are on the
        same link, but the end location is behind the start direction with
        respect to the direction of travel. If True, a vehicle is allowed to
        reverse on the link, in which case the distance will be negative, if
        False, the vehicle must "circle the block".
        :type allow_reverse: bool
        """
        self.g = g
        self.cache = defaultdict(dict)
        self.stats = CacheStats()
        self.allow_reverse = allow_reverse

    def distance(self, s_row: pd.Series, t_row: pd.Series) -> float:
        """Compute the shortest routable distance between nodes s and t

        :param s_row: The row of neighborhood to start from
        :type s_row: pd.Series
        :param t_row: The row of neighborhood to end at
        :type t_row: pd.Series
        :return: The distance travelled on the network from s to t
        :rtype: float

        Terminology:
            projection of the gps point onto the link is the point on the link
        closest to the gps ping.
            offset is the distance from the start of the link to the projection
        of the gps point onto the link

        The network distance is then the sum of the following:
        - the distance from the projection of s_pt to the end of the link
        - the shortest path distance from the end node of s_link to the start
        node of t_link
        - the distance from the start node of t_link to the projection of t_pt
        onto t_link

        EXCEPT if s_link == t_link
        This occurs when the two pings are on the same link. In this case
        return t_offset - s_offset. This will be negative if the motion is
        against the flow of traffic and allow_reverse is True. In the context of
        the transition probabilities this will be helpful to make such motion
        less likely, but still more likely than circling the block to get to an
        earlier point on the link, which will happen if allow_reverse is False.
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
        if t not in self.cache[s]:
            self._dijkstra(s, t)
            if t not in self.cache[s]:
                raise Exception(f"{t} not in {self.cache[s]}")
            self.stats.miss()
        else:
            self.stats.hit()
        return (
                (s_length - s_offset)  # distance from s pt to end of s link
                + self.cache[s][t]  # path length from s->t
                + t_offset  # distance from beginning of t link to t pt
        )

    @staticmethod
    def _weight(u, v, data) -> float:
        """Returns the length of an edge in the CRS units of the link"""
        return min(attr['geometry'].length for attr in data.values())

    def _dijkstra(self, source, target=None) -> dict:
        """Uses Dijkstra's algorithm to find shortest weighted paths

        Adapted from https://github.com/networkx/networkx/blob/f63e90ba4676fcb4ef74c5bd7ddda56be50d4c90/networkx/algorithms/shortest_paths/weighted.py#L755

        Modifications:
            - Allows only a single source
            - Removes pred and paths dicts as well as cutoff
            - Caches all distances found
            - Uses self._weight instead of passed-in weight

        Parameters
        ----------
        source : node label
            Starting node for the paths.

        target : node label, optional
            Ending node for path. Search is halted when target is found.

        Returns
        -------
        distance : dictionary
            A mapping from node to shortest distance to that node from one
            of the source nodes.

        Raises
        ------
        NodeNotFound
            If any of `sources` is not in `G`.

        Notes
        -----
        The optional predecessor and path dictionaries can be accessed by
        the caller through the original pred and paths objects passed
        as arguments. No need to explicitly return pred or paths.

        """
        g_succ = self.g._succ if self.g.is_directed() else self.g._adj
        weight = self._weight

        push = heappush
        pop = heappop
        dist = self.cache[source]  # dictionary of final distances
        seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        if source not in self.g:
            raise nx.NodeNotFound(f"Source {source} not in G")
        seen[source] = 0
        push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                break
            for u, e in g_succ[v].items():
                cost = weight(v, u, e)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if u in dist:
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        raise ValueError("Contradictory paths found:",
                                         "negative weights?")
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
        if target not in dist:
            dist[target] = np.inf
        self.cache[source] = dist
        return dist


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
