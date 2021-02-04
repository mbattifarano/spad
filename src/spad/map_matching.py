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
from typing import Union, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy


# TODO: why does the path look so bad?
def map_match(links: gpd.GeoDataFrame, trajectory: gpd.GeoDataFrame,
              shortest_path_calculator: ShortestPathCalculator,
              threshold: float = 250.0, accuracy: str = 'accuracy',
              scale: float = 1.0) -> pd.Series:
    neighborhood = link_neighborhood(links, trajectory, threshold)
    neighborhood['p_emit'] = emission_probabilities(neighborhood, accuracy)
    successors = Successors(neighborhood.groupby(
        by=trajectory.index.names,
        group_keys=False
    ))
    neg_log_likelihood = {}
    pred = defaultdict(list)
    seen = {Terminals.source: 0}
    c = count()
    fringe = []
    heappush(fringe, (0, next(c), Terminals.source))
    while fringe:
        nll, _, u = heappop(fringe)
        # u is either a Terminal or an index into neighborhood
        if u in neg_log_likelihood:
            continue  # already searched this node
        neg_log_likelihood[u] = nll
        if u is Terminals.target:
            break
        vs = successors.successors(u)
        if vs is Terminals.target:
            heappush(fringe, (nll, next(c), Terminals.target))
            pred[Terminals.target].append(u)
        else:
            gps_label, group = vs
            u_row = u if isinstance(u, Terminals) else neighborhood.loc[u]
            for v_label, row in group.iterrows():
                cost = (
                        node_cost(row)
                        + link_cost(shortest_path_calculator, u_row, row, scale)
                )
                uv_nll = neg_log_likelihood[u] + cost
                if v_label in neg_log_likelihood:
                    v_nll = neg_log_likelihood[v_label]
                    pred[v_label].append(u)
                    if uv_nll < v_nll:
                        raise ValueError(
                            "Contradictory paths found, negative weights?")
                    elif uv_nll == v_nll:
                        pred[v_label].append(u)
                elif v_label not in seen or uv_nll < seen[v_label]:
                    seen[v_label] = uv_nll
                    heappush(fringe, (uv_nll, next(c), v_label))
                    pred[v_label].append(u)
                elif uv_nll == seen[v_label]:
                    pred[v_label].append(u)
    path = []
    u = Terminals.target
    while True:
        u = pred[u][0]
        if u is Terminals.source:
            break
        path.append(u)
    return list(reversed(path)), neg_log_likelihood


class Terminals(Enum):
    source = auto()
    target = auto()


Node = Union[Terminals, pd.Series]


def link_cost(shortest_path_calculator: ShortestPathCalculator,
              u: Node, v: Node, scale: float = 1.0):
    if u is Terminals.source or v is Terminals.target:
        return 0.0
    return -np.log(
        transition_probabilities(shortest_path_calculator, u, v, scale)
    )


def node_cost(u: Node):
    if isinstance(u, Terminals):
        return 0.0
    else:
        return -np.log(u.p_emit)


class Successors:
    def __init__(self, neighbor_groups: DataFrameGroupBy):
        self.groups = []
        self.label_index = {
            Terminals.source: -1
        }
        for i, (name, group) in enumerate(neighbor_groups):
            self.groups.append((name, group))
            self.label_index[name] = i

    def successors(self, label):
        if label is Terminals.source:
            gps_index = label
        else:
            gps_index = _get_gps_key(label)
        i = self.label_index[gps_index]
        try:
            return self.groups[i + 1]
        except IndexError:
            return Terminals.target


LinkKey = Tuple[int, int, int]
_i_link = 3


def _get_link_key(compound_index: tuple) -> LinkKey:
    return compound_index[-_i_link:]


def _get_gps_key(compound_index: tuple) -> Tuple:
    return compound_index[:-_i_link]


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
    neighborhood['offset'] = neighborhood.apply(_offset_on_link, axis=1)
    return neighborhood


def _offset_on_link(row: pd.Series) -> float:
    return row.link_geometry.project(row.geometry)


def emission_probabilities(neighborhood: gpd.GeoDataFrame,
                           accuracy: Union[float, str, pd.Series]) \
        -> gpd.GeoSeries:
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
                             u_row: pd.Series, v_row: pd.Series,
                             scale: float = 1.0) -> float:
    """Compute transition probabilities between successive point-link pairs.

    :param shortest_path_calculator: An instance of ShortestPathCalculator for
    the underlying road network
    :type shortest_path_calculator: ShortestPathCalculator
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
    measurement_distance = u_row.geometry.distance(v_row.geometry)
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
    def __init__(self, g: nx.MultiDiGraph, allow_reverse: bool = True):
        self.g = g
        self.cache = defaultdict(dict)
        self.stats = CacheStats()
        self.allow_reverse = allow_reverse

    def distance(self, s_row: pd.Series, t_row: pd.Series) -> float:
        """Compute the shortest routable distance between nodes s and t

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
        against the flow of traffic. In the context of the transition
        probabilities this will be helpful to make such motion less likely, but
        still more likely than circling the block to get to an earlier point
        on the link.
        """
        s_offset = s_row.offset
        t_offset = t_row.offset
        s_link = _get_link_key(s_row.name)
        t_link = _get_link_key(t_row.name)
        if s_link == t_link:
            d = t_offset - s_offset
            if d >= 0 or self.allow_reverse:
                return d
        s_length = s_row.link_geometry.length
        s_u, s, k = s_link
        t, t_v, k = t_link
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
        """Returns the weight for a given multi-edge"""
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
    def __init__(self):
        self.hits = 0
        self.misses = 0

    @property
    def calls(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.calls

    @property
    def miss_rate(self) -> float:
        return self.misses / self.calls

    def hit(self):
        self.hits += 1

    def miss(self):
        self.misses += 1
