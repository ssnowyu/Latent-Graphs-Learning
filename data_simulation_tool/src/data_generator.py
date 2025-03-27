import math
import random
import uuid
from typing import List, Optional, Dict, Any, Tuple, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.result import Snapshot
from src.utils.data_utils import DataReader
from src.utils.time_utils import (
    MetaDistribution,
    MetaDistribution4Constant,
    MetaDistribution4Uniform,
    MetaDistribution4Gaussian,
)
from src.chain import Chain, DisType


def create_hyper_distribution(
    dis_type: List[DisType],
    params: List[List[float]],
    lower_limit: float = -float("inf"),
    upper_limit: float = float("inf"),
    params_corrected: bool = True,
) -> Dict[DisType, MetaDistribution]:
    hyper_distributions = {}
    for t, ps in zip(dis_type, params):
        if t is DisType.CONSTANT:
            hyper_distributions[DisType.CONSTANT] = MetaDistribution4Constant(
                *ps, lower_limit, upper_limit, params_corrected
            )
        if t is DisType.UNIFORM:
            hyper_distributions[DisType.UNIFORM] = MetaDistribution4Uniform(
                *ps, lower_limit, upper_limit, params_corrected
            )
        if t is DisType.GAUSSIAN:
            hyper_distributions[DisType.GAUSSIAN] = MetaDistribution4Gaussian(
                *ps, lower_limit, upper_limit, params_corrected
            )
        # if t is DistributionType.EXPONENTIAL:
        #     hyper_distributions[DistributionType.EXPONENTIAL] = MetaDistribution4Exponential(*ps, lower_limit,
        #                                                                                     upper_limit)
    return hyper_distributions


def str_to_distribution_type(s: str):
    if s == "constant":
        return DisType.CONSTANT
    if s == "uniform":
        return DisType.UNIFORM
    if s == "gaussian":
        return DisType.GAUSSIAN
    # if s == 'exponential':
    #     return DistributionType.EXPONENTIAL


def extract_unique_nodes(data: List[List[List[str]]]) -> List[str]:
    strings = set()  # 用于存储提取的字符串的集合

    for sublist in data:
        for subsublist in sublist:
            for item in subsublist:
                if isinstance(item, str):
                    strings.add(item)

    return list(strings)


class DataGenerator:
    def __init__(
        self,
        data_path: str,
        num_chain: int,
        num_step: int,
        interval_time: float,
        observed_time: float,
        chain_types: List[str],
        chain_meta_distribution_params: List[List[float]],
        chain_type_ratio: List[float],
        edge_types: List[str],
        edge_meta_distribution_params: List[List[float]],
        edge_type_ratio: List[float],
        start_steps: Dict[int, int],
        noise_ratio: float,
        min_chain_len: Optional[int] = None,
        max_chain_len: Optional[int] = None,
        min_execute_time: Optional[float] = -float("inf"),
        max_execute_time: Optional[float] = float("inf"),
        distribution_params_corrected: bool = True,
        execute_edge_chain_ratio: float = 0.5,
    ):
        """
        Args:
            data_path:
            num_chain:
            num_step:
            chain_types:
            chain_meta_distribution_params:
            chain_type_ratio:
            edge_types:
            edge_meta_distribution_params:
            edge_type_ratio:
            max_chain_len:
            min_chain_len:
            max_execute_time:
            min_execute_time:
            start_steps: start step -> number, step is from 0 to num_step - 1
        """
        # checkout parameters
        assert num_chain == sum(start_steps.values())

        self.reader = DataReader(data_path)
        self.num_chain = num_chain
        self.num_step = num_step
        self.interval_time = interval_time
        self.observed_time = observed_time
        self.step_time = interval_time + observed_time
        self.chain_types = [str_to_distribution_type(s) for s in chain_types]
        self.chain_type_ratio = chain_type_ratio
        self.edge_types = [str_to_distribution_type(s) for s in edge_types]
        assert math.isclose(sum(edge_type_ratio), 1.0, rel_tol=1e-5)
        self.edge_type_ratio = edge_type_ratio
        self.max_chain_len = max_chain_len
        self.min_chain_len = min_chain_len
        self.max_execute_time = max_execute_time
        self.min_execute_time = min_execute_time
        self.start_steps = start_steps
        self.noise_ratio = noise_ratio
        self.execute_edge_chain_ratio = execute_edge_chain_ratio

        self.normalize_ratio()

        self.chain_meta_distribution: Optional[List[MetaDistribution]] = None
        self.edge_meta_distribution: Optional[List[MetaDistribution]] = None

        # init node and chain meta distribution
        self.chain_meta_distribution = create_hyper_distribution(
            self.chain_types,
            chain_meta_distribution_params,
            lower_limit=self.min_execute_time,
            upper_limit=self.max_execute_time,
            params_corrected=distribution_params_corrected,
        )
        self.edge_meta_distribution = create_hyper_distribution(
            self.edge_types,
            edge_meta_distribution_params,
            lower_limit=self.min_execute_time,
            upper_limit=self.max_execute_time,
            params_corrected=distribution_params_corrected,
        )

        # instances of chain
        self.chains: Optional[List[Chain]] = None
        self.snapshots: Optional[List[Snapshot]] = None
        self.global_graph: Optional[nx.DiGraph] = None
        # self.nodes: Optional[Dict[str, Dict[str, Any]]] = None

    def generate_node_distribution(self, node_type: DisType):
        return self.edge_meta_distribution[node_type].generate()

    def generate_chain_distribution(self, chain_type: DisType):
        return self.chain_meta_distribution[chain_type].generate()

    def sample_and_set_chains(self):
        nodes, edges = self.reader.select_traces_and_get_nodes_and_edges(
            self.num_chain, self.min_chain_len, self.max_chain_len
        )
        self._create_and_set_chains(nodes, edges)

    def _create_and_set_chains(
        self,
        nodes: List[List[List[str]]],
        edges: List[List[Dict[Tuple[str, str], str]]],
    ):
        # init self.chains
        self.chains = []
        for nodes_for_a_chain, edges_for_a_chain in zip(nodes, edges):
            self._create_and_set_a_chain(nodes_for_a_chain, edges_for_a_chain)

    def _create_and_set_a_chain(
        self,
        nodes_for_chains: Sequence[Sequence[str]],
        edges_for_chains: Sequence[Dict[Tuple[str, str], str]],
    ):
        chains = []
        for nodes, edges in zip(nodes_for_chains, edges_for_chains):
            chain_type = random.choice(self.chain_types)
            c = Chain(
                chain_type=chain_type,
                time_dis=self.generate_chain_distribution(chain_type),
            )
            # c.add_nodes_from(nodes)
            edge_dis_types = np.random.choice(
                self.edge_types, size=len(edges), replace=True, p=self.edge_type_ratio
            )
            edge_dis = [self.generate_node_distribution(t) for t in edge_dis_types]
            us, vs = zip(*edges.keys())
            e_ids = edges.values()
            c.add_edges_from(us, vs, e_ids, edge_dis_types, edge_dis)
            assert all(elem in c.nodes for elem in nodes)
            c.add_nodes_from(nodes)
            # execute chain
            c.execute(self.execute_edge_chain_ratio, start_time=0.0)
            chains.append(c)

        assert len(chains) > 0
        # merge all chains
        c = chains[0]
        if len(chains) > 1:
            for c_for_appending in chains[1:]:
                c.append_(c_for_appending)
        self.chains.append(c)

        source_nodes, target_nodes = zip(*c.edges)
        assert all(elem in source_nodes or elem in target_nodes for elem in c.nodes)

    def observe(self):
        self.snapshots = [Snapshot(i, i * self.step_time, self.observed_time) for i in range(self.num_step)]
        for c in self.chains:
            for each_snapshot in self.snapshots:
                each_snapshot.add_nodes_from(c.nodes)

            for u, v, attr in c.edges(data=True):
                timestamp = attr["timestamp"]
                snapshot_index = math.floor(timestamp / self.step_time)
                if snapshot_index < self.num_step and timestamp % self.step_time <= self.observed_time:
                    self.snapshots[snapshot_index].add_edge(u, v)

        self.global_graph = nx.DiGraph()
        for c in self.chains:
            # print(nx.isolates(c.g))
            # self.global_graph.add_nodes_from(c.nodes)
            self.global_graph.add_edges_from(c.edges)
            source_nodes, target_nodes = zip(*c.edges)
            assert all(elem in source_nodes or elem in target_nodes for elem in c.nodes)

    def normalize_ratio(self):
        edge_ratio_sum = sum(self.edge_type_ratio)
        if not math.isclose(edge_ratio_sum, 1.0):
            self.edge_type_ratio = [i / edge_ratio_sum for i in self.edge_type_ratio]
        chain_ratio_sum = sum(self.chain_type_ratio)
        if not math.isclose(chain_ratio_sum, 1.0):
            self.chain_type_ratio = [i / chain_ratio_sum for i in self.chain_type_ratio]

    def set_start_time_for_chains(self):
        start_step_indices = []
        for start_step, num in self.start_steps.items():
            start_step_indices.extend([start_step for _ in range(num)])
        random.shuffle(start_step_indices)
        for chain, start_step in zip(self.chains, start_step_indices):
            start_time = start_step * self.step_time + np.random.uniform(
                0.0, self.step_time
            )
            chain.set_start_time(start_time)

    def add_noisy_edges(self):
        for each_snapshot in self.snapshots:
            num_noisy_edges = int(len(each_snapshot.edges) * self.noise_ratio)

            assert all(elem in self.global_graph.nodes for elem in each_snapshot.nodes)

            edges_for_completed_graph = []
            for u in each_snapshot.nodes:
                for v in each_snapshot.nodes:
                    edges_for_completed_graph.append((u, v))
            unconnected_edges = set(edges_for_completed_graph) - set(
                each_snapshot.edges
            )

            if len(unconnected_edges) <= num_noisy_edges:
                each_snapshot.add_edges_from(unconnected_edges)
                break

            each_snapshot.add_edges_from(
                random.sample(list(unconnected_edges), num_noisy_edges)
            )

            assert all(elem in self.global_graph.nodes for elem in each_snapshot.nodes)

    def generate_data(self):
        print("sample and set chains")
        self.sample_and_set_chains()
        print("set start time for chains")
        self.set_start_time_for_chains()
        print("observing")
        self.observe()
        print("add noisy edges")
        self.add_noisy_edges()

    def to_dataframe(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        sample_id = uuid.uuid1().hex
        print(sample_id)

        # chain
        chain_data = [
            (sample_id, idx, edge[0], edge[1])
            for idx, chain in enumerate(self.chains)
            for edge in chain.edges
        ]
        chain_df = pd.DataFrame(
            chain_data, columns=["sample_id", "chain_id", "source", "target"]
        )

        # snapshots
        snapshot_data = [
            (
                sample_id,
                snapshot.time_step_index,
                snapshot.start_time,
                snapshot.end_time,
                edge[0],
                edge[1],
            )
            for snapshot in self.snapshots
            for edge in snapshot.edges
        ]
        snapshot_df = pd.DataFrame(
            snapshot_data,
            columns=[
                "sample_id",
                "time_step",
                "start_time",
                "end_time",
                "source",
                "target",
            ],
        )

        # global graph
        graph_data = [(sample_id, edge[0], edge[1]) for edge in self.global_graph.edges]

        global_graph_df = pd.DataFrame(
            graph_data, columns=["sample_id", "source", "target"]
        )

        return chain_df, snapshot_df, global_graph_df


if __name__ == "__main__":
    chains = pd.read_csv("../data/result/chains.csv")
    snapshots = pd.read_csv("../data/result/snapshots.csv")
