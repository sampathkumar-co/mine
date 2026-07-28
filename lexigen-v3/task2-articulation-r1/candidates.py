from __future__ import annotations

from typing import Callable, TypedDict

import igraph as ig
import networkx as nx
import rustworkx as rx


class Problem(TypedDict):
    num_nodes: int
    edges: list[list[int]]


class Solution(TypedDict):
    articulation_points: list[int]


def rustworkx_native(problem: Problem) -> Solution:
    graph = rx.PyGraph()
    graph.add_nodes_from([None] * problem["num_nodes"])
    graph.add_edges_from_no_data(problem["edges"])
    points = sorted(int(node) for node in rx.articulation_points(graph))
    return {"articulation_points": points}


def igraph_native(problem: Problem) -> Solution:
    graph = ig.Graph(
        n=problem["num_nodes"],
        edges=problem["edges"],
        directed=False,
    )
    points = sorted(int(node) for node in graph.articulation_points())
    return {"articulation_points": points}


def networkx_control(problem: Problem) -> Solution:
    graph = nx.Graph()
    graph.add_nodes_from(range(problem["num_nodes"]))
    graph.add_edges_from(problem["edges"])
    points = sorted(int(node) for node in nx.articulation_points(graph))
    return {"articulation_points": points}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "rustworkx_native": rustworkx_native,
    "igraph_native": igraph_native,
    "networkx_control": networkx_control,
}
