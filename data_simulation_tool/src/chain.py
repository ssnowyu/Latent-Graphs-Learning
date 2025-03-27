import uuid
from enum import Enum
from functools import cached_property
from typing import List, Any, Iterable, Optional, Union, Sequence, Tuple, Dict

import networkx as nx

from src.utils.time_utils import Distribution


class DisType(Enum):
    # Distribution Type
    CONSTANT = 0
    UNIFORM = 1
    GAUSSIAN = 2
    EXPONENTIAL = 3


def get_prefix(edge_id: str):
    if "." not in edge_id:
        return []
    prefixes = []
    parts = edge_id.split(".")
    prefix = ""
    for part in parts:
        prefix += part
        prefixes.append(prefix)
        prefix += "."
    return prefixes[:-1]


class Chain:
    def __init__(
        self,
        chain_type: Optional[DisType] = None,
        time_dis: Optional[Distribution] = None,
        # edges: List[Tuple[str, str]] = None,
        # edge_ids: List[str] = None,
        # edge_dis_types: List[Distribution] = None,
    ):
        self.g: nx.DiGraph = nx.DiGraph()
        self.chain_type: Optional[DisType] = chain_type
        self.time_dis: Optional[Distribution] = time_dis
        self.id = uuid.uuid1().hex
        self.id_edge: Optional[Dict[str, Tuple[str, str]]] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __getattr__(self, item):
        getattr(self.g, item)

    @cached_property
    def nodes(self):
        return self.g.nodes

    @cached_property
    def edges(self):
        return self.g.edges

    def add_node(self, node_for_adding: Any, **attr):
        self.g.add_node(node_for_adding, **attr)

    def add_nodes_from(self, nodes_for_adding: Any, **attr: Any):
        self.g.add_nodes_from(nodes_for_adding, **attr)

    def add_edge(
        self,
        u_for_edge: Any,
        v_for_edge: Any,
        e_id: str,
        edge_dis_type: DisType,
        edge_dis: Distribution,
        timestamp: Optional[float] = None,
        **attr: Any
    ):
        if timestamp is None:
            self.g.add_edge(
                u_for_edge,
                v_for_edge,
                id=e_id,
                dis_type=edge_dis_type,
                dis=edge_dis,
                **attr
            )
            return

        self.g.add_edge(
            u_for_edge,
            v_for_edge,
            id=e_id,
            dis_type=edge_dis_type,
            dis=edge_dis,
            timestamp=timestamp,
            **attr
        )

    def add_edges_from(
        self,
        us_for_edges: List[Any],
        vs_for_edges: List[Any],
        e_ids: List[str],
        edge_dis_types: List[DisType],
        edge_dis: List[Distribution],
        edge_timestamps: Optional[List[float]] = None,
    ):
        if edge_timestamps is None:
            self.g.add_edges_from(
                [
                    (u, v, {"id": e_id, "dis_type": dis_type, "dis": dis})
                    for u, v, e_id, dis_type, dis in zip(
                        us_for_edges, vs_for_edges, e_ids, edge_dis_types, edge_dis
                    )
                ]
            )
            return

        self.g.add_edges_from(
            [
                (u, v, {"id": e_id, "dis_type": dis_type, "dis": dis, "timestamp": t})
                for u, v, e_id, dis_type, dis, t in zip(
                    us_for_edges,
                    vs_for_edges,
                    e_ids,
                    edge_dis_types,
                    edge_dis,
                    edge_timestamps,
                )
            ]
        )

    def _generate_time_for_edge(self, edge: Tuple[str, str], time_ratio):
        u, v = edge
        return (
            time_ratio * self.time_dis.restrictively_generate()
            + (1.0 - time_ratio) * self.edges[u, v]["dis"].restrictively_generate()
        )

    def _set_timestamp(self, edge: Tuple[str, str], timestamp: float):
        self.g.edges[edge[0], edge[1]]["timestamp"] = timestamp

    def get_timestamp_by_id(self, edge_id: str):
        edge = self.id_edge[edge_id]
        u, v = edge
        return self.g.edges[u, v]["timestamp"]

    def execute(self, time_ratio: float = 0.5, start_time: float = 0.0):
        id_edge = {self.g.edges[u, v]["id"]: (u, v) for u, v in self.g.edges}
        # id_edge = {}
        # for u, v, attr in self.g.edges(data=True):
        #     if attr["id"] not in id_edge:
        #         id_edge[attr["id"]] = [(u, v)]
        #     else:
        #         id_edge

        self.id_edge = {
            k: v
            for k, v in sorted(
                id_edge.items(),
                key=lambda x: len(x[0].split(".")) if "." in x[0] else 1,
            )
        }
        for e_id, e in self.id_edge.items():
            if "." not in e_id:
                # initial edge
                self._set_timestamp(edge=e, timestamp=start_time)
                continue
            pre_edge_ids = get_prefix(edge_id=e_id)
            legal_pre_edge_ies = [
                item for item in pre_edge_ids if item in self.id_edge.keys()
            ]
            pre_edge_timestamps = [
                self.get_timestamp_by_id(item) for item in legal_pre_edge_ies
            ]
            timestamp_for_pre_edge = (
                max(pre_edge_timestamps) if len(pre_edge_timestamps) > 0 else 0.0
            )
            timestamp = timestamp_for_pre_edge + self._generate_time_for_edge(
                edge=e, time_ratio=time_ratio
            )
            self._set_timestamp(e, timestamp=timestamp)

        # set the start and end time for chain
        self.start_time = start_time

        # for u, v in self.g.edges:
        #     pass

        self.end_time = max([self.g.edges[u, v]["timestamp"] for u, v in self.g.edges])

    def set_start_time(self, start_time: float):
        offset_time = start_time - self.start_time
        self.start_time = start_time
        for u, v in self.g.edges:
            self.g.edges[u, v]["timestamp"] += offset_time

    def offset_time(self, off: float):
        self.start_time += off
        for u, v in self.g.edges:
            self.g.edges[u, v]["timestamp"] += off

    def append_(self, b: "Chain"):
        # This operation must be followed by executing
        # The default is to put the timestamp of b after self
        a_tail_u, a_tail_v, end_timestamp_for_a = max(
            [(u, v, self.edges[u, v]["timestamp"]) for u, v in self.edges],
            key=lambda x: x[2],
        )
        b_head_u, b_head_v, attr = min(
            [(u, v, attr) for u, v, attr in b.edges(data=True)],
            key=lambda x: x[-1]["timestamp"],
        )
        time_for_adding_edge = self.edges[a_tail_u, a_tail_v]["dis"].restrictively_generate()
        time_for_fist_edge_of_b = b.edges[b_head_u, b_head_v]['dis'].restrictively_generate()
        b.offset_time(end_timestamp_for_a + time_for_adding_edge + time_for_fist_edge_of_b)
        # Add an edge to connect self and b
        self.add_edge(
            a_tail_v,
            b_head_u,
            uuid.uuid1().hex,
            attr["dis_type"],
            attr["dis"],
            self.edges[a_tail_u, a_tail_v]['timestamp'] + time_for_adding_edge,
        )
        # Add the edges in b to self and delete the duplicate edges in b and self
        for u, v, attr in b.edges(data=True):
            if (u, v) not in self.edges:
                self.add_edge(
                    u, v, attr["id"], attr["dis_type"], attr["dis"], attr["timestamp"]
                )
