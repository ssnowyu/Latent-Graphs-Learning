from typing import Sequence

import networkx as nx
from matplotlib import pyplot as plt


def get_root_node(g: nx.DiGraph):
    nodes_with_zero_indegree = [node for node, indegree in g.in_degree if indegree == 0]
    return nodes_with_zero_indegree[0] if len(nodes_with_zero_indegree) == 1 else None


def splice_graphs(graphs: Sequence[nx.DiGraph]) -> nx.DiGraph:
    """
    splice given graphs assuming they have no the same node.
    :param graphs:
    :return:
    """
    assert len(graphs) > 1, "the number of graphs must > 1"
    nodes_with_zero_indegree = [
        node for node, indegree in graphs[0].in_degree if indegree == 0
    ]
    assert len(nodes_with_zero_indegree) == 1

    nodes = []
    for ns in [g.nodes for g in graphs]:
        nodes.extend(ns)
    assert len(nodes) == len(set(nodes))

    new_g = nx.DiGraph()
    for g in graphs:
        path = nx.dag_longest_path(new_g)
        if len(path) == 0:
            new_g = nx.compose(new_g, g)
            continue
        source = path[-1]
        target = get_root_node(g)
        new_g = nx.compose(new_g, g)
        new_g.add_edge(u_of_edge=source, v_of_edge=target)

    return new_g
