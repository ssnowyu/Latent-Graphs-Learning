from functools import cached_property
from typing import Any

import networkx as nx


class Snapshot:
    # def __init__(self, time_step_index: int, step_time: float):
    #     self.time_step_index = time_step_index
    #     self.step_time = step_time
    #     self.start_time = time_step_index * step_time
    #     self.end_time = self.start_time + step_time
    #     self.g = nx.DiGraph()

    def __init__(self, index:int, start_time: float, duration: float):
        self.time_step_index = index
        self.start_time = start_time
        self.end_time = start_time + duration
        self.g = nx.DiGraph()

    @cached_property
    def nodes(self):
        return self.g.nodes

    @cached_property
    def edges(self):
        return self.g.edges

    def add_node(self, node_for_adding: Any, **attr: Any):
        self.g.add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding: Any, **attr: Any):
        self.g.add_nodes_from(nodes_for_adding, **attr)

    def add_edge(self, source: Any, target: Any, **attr: Any):
        self.g.add_edge(source, target, **attr)

    def add_edges_from(self, edges_for_adding: Any, **attr: Any):
        self.g.add_edges_from(edges_for_adding, **attr)
