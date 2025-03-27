import uuid
from typing import Optional, Tuple, Sequence, Dict, Union, List
import networkx as nx
import pandas as pd
import py2neo
import yaml

from src.result import Snapshot
from src.value_chain import Chain

config_path = "../../configs/neo4j/neo4j.yaml"


class Neo4jUtils:
    def __init__(self):
        # load config
        self.g: Optional[py2neo.Graph] = None

        self.database_url: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None

    def load_config(self):
        with open(config_path, "r", encoding="utf-8") as f:
            params_str = f.read()
            print(f"load neo4j config from {config_path}")
            print("=====================================================")
            print(f"{params_str}\n")
            params = yaml.load(params_str, Loader=yaml.FullLoader)
            self.database_url = params["url"]
            self.username = params["username"]
            self.password = params["password"]

    def connect(
        self, database_url: Optional[str] = None, auth: Optional[Tuple[str, str]] = None
    ):
        if database_url is None or auth is None:
            if (
                self.database_url is not None
                and self.username is not None
                and self.password is not None
            ):
                self.g = py2neo.Graph(
                    self.database_url, auth=(self.username, self.password), name="neo4j"
                )
                return
            else:
                raise ValueError(
                    "provide url and auth in connecting or load configs before connecting."
                )
        self.g = py2neo.Graph(database_url, auth=auth, name="neo4j")

    # def upload_chains(
    #     self, chains: Sequence[ValueChain], sample_id: Optional[str] = None
    # ):
    #     print("upload chains")
    #     if sample_id is None:
    #         sample_id = ""
    #
    #     for chain in chains:
    #         chain_id = uuid.uuid1().hex
    #         nodes = {
    #             n: py2neo.Node("chain", sample_id=sample_id, chain_id=chain_id, name=n)
    #             for n in list(chain.nodes)
    #         }
    #         for n in nodes.values():
    #             self.g.create(n)
    #         for e in chain.edges:
    #             edge = py2neo.Relationship(nodes[e[0]], nodes[e[1]])
    #             self.g.create(edge)
    #
    # def upload_snapshots(
    #     self, snapshots: Sequence[Snapshot], sample_id: Optional[str] = None
    # ):
    #     print("upload snapshots")
    #     if sample_id is None:
    #         sample_id = ""
    #
    #     for idx, each_snapshot in enumerate(snapshots):
    #         nodes = {
    #             n: py2neo.Node(
    #                 "snapshot",
    #                 str(each_snapshot.nodes[n]["chain_id"]),
    #                 sample_id=sample_id,
    #                 step=idx,
    #                 name=n,
    #             )
    #             for n in list(each_snapshot.nodes)
    #         }
    #         for n in nodes.values():
    #             self.g.create(n)
    #         for e in each_snapshot.edges:
    #             edge = py2neo.Relationship(nodes[e[0]], nodes[e[1]])
    #             self.g.create(edge)
    #
    # def upload_global_graph(self, graph: nx.DiGraph, sample_id: Optional[str] = None):
    #     print("upload global graph")
    #     nodes = {
    #         n: py2neo.Node("global", sample_id=sample_id, name=n)
    #         for n in list(graph.nodes)
    #     }
    #     for n in nodes.values():
    #         self.g.create(n)
    #     for e in graph.edges:
    #         edge = py2neo.Relationship(nodes[e[0]], nodes[e[1]])
    #         self.g.create(edge)

    def upload_chains_from_csv(self, path: str):
        print(f"upload chains from {path}")
        df = pd.read_csv(path)
        for sample_id in df["sample_id"].unique():
            sample_df = df[df["sample_id"] == sample_id]
            nodes: Dict[str, Dict[str, Union[str, List[str]]]] = {}
            for chain_id in sample_df["chain_id"].unique():
                chain_df = sample_df[sample_df["chain_id"] == chain_id]
                for index, row in chain_df.iterrows():
                    s_t_nodes = (row["source"], row["target"])
                    for n in s_t_nodes:
                        if n in nodes:
                            if row["chain_id"] not in nodes[n]["chain_ids"]:
                                nodes[n]["chain_ids"].append(row["chain_id"])
                        else:
                            nodes[n] = {"chain_ids": [row["chain_id"]]}
            nodes = {
                n: py2neo.Node(
                    "chain", sample_id=sample_id, name=n, chain_ids=attr["chain_ids"]
                )
                for n, attr in nodes.items()
            }
            for n in nodes.values():
                self.g.create(n)
            for index, row in sample_df.iterrows():
                edge = py2neo.Relationship(nodes[row["source"]], nodes[row["target"]])
                self.g.create(edge)
        print("complete")


if __name__ == "__main__":
    neo = Neo4jUtils()
    neo.load_config()
    neo.connect()
    neo.upload_chains_from_csv("../../data/result/chains.csv")
