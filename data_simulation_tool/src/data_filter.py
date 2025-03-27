import networkx as nx
import pandas as pd
import multiprocessing


class DataFilter:
    def __init__(self, path: str):
        print(f"load data from {path}")
        data: pd.DataFrame = pd.read_csv(path, index_col="Unnamed: 0")
        # data = data[:1000]
        self.data = data.sort_values(by="timestamp")
        self.remove_null()
        self.remove_duplicate_edges()
        self.remove_replies_of_rpc_and_http()
        # self.remove_construct_error()
        self.set_len()

    def remove_null(self):
        print("remove nul")
        self.data = self.data.dropna(
            axis=0,
            how="any",
            subset=["traceid", "timestamp", "rpcid", "um", "rpctype", "dm", "rt"],
        )
        values_to_filter = ["", "(?)"]
        self.data = self.data[~self.data.isin(values_to_filter).any(axis=1)]

    def remove_duplicate_edges(self):
        print("remove duplicate edges")
        self.data = self.data.drop_duplicates(
            subset=["traceid", "um", "dm"], keep="first"
        )

    def remove_replies_of_rpc_and_http(self):
        print("remove replies of rpc and http")
        self.data = self.data.drop_duplicates(subset=["traceid", "rpcid"], keep="first")
        self.data = self.data[self.data["rt"] >= 0]

    # def remove_duplicate_rpcid(self):
    #     print("remove duplicate traceid")
    #     self.data = self.data.drop_duplicates(subset=["traceid", ])

    def remove_construct_error(self):
        candidate_trace = []
        unique_trace_id = self.data["traceid"].unique()
        for trace_id in unique_trace_id:
            df = self.data[self.data["traceid"] == trace_id]
            ums = df["um"].tolist()
            dms = df["dm"].tolist()
            # remove graph with circle like (a -> b), (b -> a)
            edges = list(ums) + list(dms)
            if len(edges) != len(set(edges)):
                continue
            g = nx.DiGraph()
            g.add_edges_from(zip(ums, dms))
            # remove graph with circle
            if len(nx.cycle_basis(g.to_undirected())):
                continue
            # remove graph with multi-root
            nodes_with_zero_indegree = [
                node for node, indegree in g.in_degree if indegree == 0
            ]
            if len(nodes_with_zero_indegree) != 1:
                continue
            if not nx.is_weakly_connected(g):
                continue
            candidate_trace.append(trace_id)
        self.data = self.data.loc[self.data["traceid"].isin(candidate_trace)]

    # def set_len(self):
    #     self.data["len"] = [0 for _ in range(len(self.data))]
    #     for t_id in self.data["traceid"].unique():
    #         local_df = self.data[self.data["traceid"] == t_id]
    #         ums = local_df["um"]
    #         dms = local_df["dm"]
    #         g = nx.DiGraph()
    #         g.add_edges_from(zip(ums, dms))
    #         print(t_id)
    #         length = nx.dag_longest_path_length(g)
    #         self.data.loc[self.data["traceid"] == t_id, "len"] = length

    def set_len(self):
        print("set length")
        self.data["len"] = [0 for _ in range(len(self.data))]
        for t_id in self.data["traceid"].unique():
            local_df = self.data[self.data["traceid"] == t_id]["rpcid"]
            lengths = [len(rpc.split(".") if "." in rpc else 1) for rpc in local_df]
            self.data.loc[self.data["traceid"] == t_id, "len"] = max(lengths)

    def save(self, path: str):
        print(f"save data to {path}")
        self.data.to_csv(path, index=False)


def filter_data(i: int):
    path = f"../data/raw_call_graph/MSCallGraph_{i}.csv"
    data_filter = DataFilter(path)
    data_filter.save(f"../data/raw/call_graph_{i}.csv")


if __name__ == "__main__":
    processes = []  # 存储进程对象的列表

    for i in range(5):
        p = multiprocessing.Process(target=filter_data, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("所有进程已完成")
