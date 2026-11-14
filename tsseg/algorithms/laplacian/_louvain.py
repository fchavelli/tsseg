# Adapted from NetworkX 3.5 (``networkx/algorithms/community/louvain.py`` and
# ``modularity`` in ``networkx/algorithms/community/quality.py``): the
# undirected, weighted case only, on a minimal adjacency structure, without
# the networkx dependency.
#
# Copyright (c) 2004-2025, NetworkX Developers
# Aric Hagberg <hagberg@lanl.gov>
# Dan Schult <dschult@colgate.edu>
# Pieter Swart <swart@lanl.gov>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#
#   * Neither the name of the NetworkX Developers nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Louvain community detection (Blondel et al. 2008) on a weighted graph.

A port of ``networkx.community.louvain_communities`` for undirected weighted
graphs. The graph is a dict of dicts whose insertion order follows the
networkx ``Graph`` built from the same edge list, and the moves, gains and
random shuffles are those of networkx, so that a given seed gives the same
communities as ``networkx.community.louvain_communities(G, seed=seed)``.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

__all__ = ["louvain_communities", "louvain_from_affinity"]


class _Graph:
    """Undirected weighted graph: ``adj[u][v]`` is the shared edge dict."""

    def __init__(self) -> None:
        self.adj: dict = {}
        self.nodes: dict = {}

    def add_node(self, u, **attr) -> None:
        if u not in self.adj:
            self.adj[u] = {}
            self.nodes[u] = {}
        self.nodes[u].update(attr)

    def add_edge(self, u, v, weight: float) -> None:
        for node in (u, v):
            if node not in self.adj:
                self.adj[node] = {}
                self.nodes[node] = {}
        data = self.adj[u].get(v, {})
        data["weight"] = weight
        self.adj[u][v] = data
        self.adj[v][u] = data

    def edges(self):
        """Each edge once, as ``networkx.Graph.edges(data=True)`` lists them."""
        seen = set()
        for u, nbrs in self.adj.items():
            for v, data in nbrs.items():
                if v not in seen:
                    yield u, v, data["weight"]
            seen.add(u)

    def degree(self, u) -> float:
        nbrs = self.adj[u]
        # a self-loop counts twice, as in networkx
        return sum(d["weight"] for d in nbrs.values()) + (
            u in nbrs and nbrs[u]["weight"]
        )

    def degrees(self) -> dict:
        return {u: self.degree(u) for u in self.adj}


def _modularity(G: _Graph, communities, resolution: float) -> float:
    degree = G.degrees()
    deg_sum = sum(degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum**2

    def incident_edges(comm):
        """Edges with an end in ``comm``, each once (``G.edges(comm)``)."""
        seen = set()
        for u in comm:
            for v, data in G.adj[u].items():
                if v not in seen:
                    yield v, data["weight"]
            seen.add(u)

    def contribution(community) -> float:
        comm = set(community)
        L_c = sum(wt for v, wt in incident_edges(comm) if v in comm)
        degree_sum = sum(degree[u] for u in comm)
        return L_c / m - resolution * degree_sum * degree_sum * norm

    return sum(map(contribution, communities))


def _neighbor_weights(nbrs, node2com):
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _one_level(G: _Graph, m, partition, resolution, seed):
    node2com = {u: i for i, u in enumerate(G.adj)}
    inner_partition = [{u} for u in G.adj]
    degrees = G.degrees()
    Stot = list(degrees.values())
    nbrs = {u: {v: d["weight"] for v, d in G.adj[u].items() if v != u} for u in G.adj}
    rand_nodes = list(G.adj)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            degree = degrees[u]
            Stot[best_com] -= degree
            remove_cost = -weights2com[best_com] / m + resolution * (
                Stot[best_com] * degree
            ) / (2 * m**2)
            for nbr_com, wt in weights2com.items():
                gain = (
                    remove_cost
                    + wt / m
                    - resolution * (Stot[nbr_com] * degree) / (2 * m**2)
                )
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            Stot[best_com] += degree
            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _gen_graph(G: _Graph, partition) -> _Graph:
    H = _Graph()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)
    for node1, node2, wt in G.edges():
        com1, com2 = node2com[node1], node2com[node2]
        temp = H.adj[com1][com2]["weight"] if com2 in H.adj[com1] else 0
        H.add_edge(com1, com2, weight=wt + temp)
    return H


def louvain_communities(
    edges,
    *,
    resolution: float = 1.0,
    threshold: float = 1e-7,
    seed: int | None = None,
) -> list[set]:
    """Louvain communities of the graph given by its weighted edge list.

    Parameters
    ----------
    edges : iterable of (u, v, weight)
        Undirected edges, in the order networkx would receive them from
        ``Graph.add_weighted_edges_from``; nodes are those of the edges.
    resolution : float, default=1.0
        Resolution of the modularity.
    threshold : float, default=1e-7
        Smallest modularity gain for another aggregation level.
    seed : int or None, default=None
        Seed of the ``random.Random`` instance that shuffles the nodes.

    Returns
    -------
    list of set
        The communities of the last level (empty list for an empty graph).
    """
    rng = random.Random(seed)
    G0 = _Graph()
    for u, v, w in edges:
        G0.add_edge(u, v, weight=w)
    if not G0.adj:
        return []
    partition = [{u} for u in G0.adj]
    if not any(G0.adj[u] for u in G0.adj):
        return partition
    mod = _modularity(G0, partition, resolution)
    # networkx copies the graph before the first level: same nodes, edges
    # re-inserted in ``G.edges`` order, which can change the adjacency order
    graph = _Graph()
    for u in G0.adj:
        graph.add_node(u)
    for u, v, w in G0.edges():
        graph.add_edge(u, v, weight=w)
    m = sum(graph.degrees().values()) / 2
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, resolution, rng
    )
    improvement = True
    while improvement:
        result = [s.copy() for s in partition]
        new_mod = _modularity(graph, inner_partition, resolution)
        if new_mod - mod <= threshold:
            return result
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, resolution, rng
        )
    return [s.copy() for s in partition]


def louvain_from_affinity(
    A: np.ndarray, *, resolution: float = 1.0, seed: int | None = None
) -> list[set]:
    """Louvain communities of the graph of a symmetric affinity matrix.

    Edges are the non-zero entries above the diagonal, in row-major order
    (``np.nonzero(np.triu(A, 1))``); frames without any edge are not nodes.
    """
    i, j = np.nonzero(np.triu(A, 1))
    w = A[i, j]
    return louvain_communities(
        zip(i.tolist(), j.tolist(), w.tolist(), strict=True),
        resolution=resolution,
        seed=seed,
    )
