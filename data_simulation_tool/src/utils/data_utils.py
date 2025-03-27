import copy
import math
import random
from typing import Optional, Dict, List, Tuple, Sequence, Union, Any

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils import graph_utils, math_utils
from src.chain import Chain


class DataReader:
    def __init__(self, path: str):
        self.trace_df: pd.DataFrame = pd.read_csv(path)
        # self.pre_process()
        df = self.trace_df[["traceid", "len"]].drop_duplicates()
        self.candidate_traces: Dict[int, List[str]] = (
            df.groupby("len")["traceid"].apply(list).to_dict()
        )

    def _select_traces_for_compose_by_target_len(
        self, target_len: int
    ) -> Optional[Sequence[str]]:
        params = [(key, len(value)) for key, value in self.candidate_traces.items()]
        candidate_values, max_usage = zip(*params)
        candidate_len = math_utils.decompose(
            target_sum=target_len,
            candidate_values=candidate_values,
            max_usage=max_usage,
        )
        res = self._select_traces_by_candidate_lens(candidate_len)
        return res

    def _select_traces_by_candidate_lens(self, candidate_lens: Sequence[int]):
        selected_traces = []
        for l in candidate_lens:
            trace = random.choice(self.candidate_traces[l])
            selected_traces.append(trace)
            # remove trace has been selected
            self.candidate_traces[l].remove(trace)
        return selected_traces

    def _select_traces(
        self, n: int, min_len: Optional[int] = 0, max_len: Optional[int] = None
    ) -> Sequence[Sequence[str]]:
        assert max_len is not None, "The max_len is necessary."
        trace_len = [
            math.floor(np.random.uniform(low=min_len, high=max_len) + 1)
            for _ in range(n)
        ]
        res = [
            self._select_traces_for_compose_by_target_len(t_len) for t_len in trace_len
        ]
        return res

    # def _get_nodes_by_trace_id(self, traceid: str) -> Sequence[str]:
    #     df = self.trace_df[self.trace_df["traceid"] == traceid]
    #     ums = df["um"].tolist()
    #     dms = df["dm"].tolist()
    #     return list(set(ums + dms))

    def _get_nodes_and_edges_by_trace_id(
        self, traceid: str
    ) -> Tuple[List[str], Dict[Tuple[str, str], str]]:
        df = self.trace_df[self.trace_df["traceid"] == traceid]
        # ums = df["um"].tolist()
        # dms = df["dm"].tolist()
        # nodes = list(set(ums + dms))
        nodes = set()
        edges: Dict[Tuple[str, str], str] = {}
        for _, row in df.iterrows():
            u = row["um"]
            v = row["dm"]
            nodes.add(u)
            nodes.add(v)
            edges[(u, v)] = row["rpcid"]

        return list(nodes), edges

    # def select_traces_and_get_chains(
    #     self, n: int, min_len: Optional[int] = 0, max_len: Optional[int] = None
    # ) -> Tuple[List[List[List[str]]], List[List[List[Tuple[str, str]]]]]:
    #     assert max_len is not None, "The max_len is necessary."
    #     trace_ids: Sequence[Sequence[str]] = self._select_traces(n, min_len, max_len)
    #
    #     for each_trace_ids in trace_ids:
    #         for t_id in each_trace_ids:
    #             nodes, edges = self._get_nodes_and_edges_by_trace_id(t_id)
    #             chain = Chain()

    def select_traces_and_get_nodes_and_edges(
        self, n: int, min_len: Optional[int] = 0, max_len: Optional[int] = None
    ) -> Tuple[List[List[List[str]]], List[List[Dict[Tuple[str, str], str]]]]:
        assert max_len is not None, "The max_len is necessary."
        trace_ids = self._select_traces(n, min_len, max_len)
        nodes = []
        edges = []
        for trace_ids_for_one_trace in trace_ids:
            nodes_for_a_trace = []
            edges_for_a_trace = []
            for t_id in trace_ids_for_one_trace:
                ns, es = self._get_nodes_and_edges_by_trace_id(t_id)
                nodes_for_a_trace.append(ns)
                edges_for_a_trace.append(es)
            nodes.append(nodes_for_a_trace)
            edges.append(edges_for_a_trace)

        return nodes, edges


if __name__ == "__main__":
    r = DataReader("../../data/raw/call_graph_0.csv")
    nodes, edges = r.select_traces_and_get_nodes_and_edges(10, min_len=5, max_len=50)
    print(nodes)
    print(edges)
