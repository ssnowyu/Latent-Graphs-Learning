import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import os
import pandas as pd
import torch
import random
from tqdm import tqdm
import dgl
from src.utils.seed_utils import seed_everything
from src.utils.time_utils import rfc3339_to_unix_milliseconds

RAW_DIR = "data/Diginetica/raw"
PROCESSED_DIR = "data/Diginetica/processed"
CLUSTER_FILE = "data/Diginetica/raw/cluster.pkl"

MIN_NUM_NODES = 10
MAX_NUM_NODES = 20
NUM_EACH_LENGTH = 9000
NUM_SESSION_PER_SAMPLE = 25
NUM_SAMPLES = 2000

SAMPLE_RATIO = 0.1

cosine = torch.nn.CosineSimilarity()


def calculate_similar_sessions(x: torch.Tensor, ys: torch.Tensor, top_k: int):
    similarity = cosine(x.unsqueeze(0), ys)
    _, top_k_indices = torch.topk(similarity, top_k)
    return top_k_indices.tolist()


def run():
    df = pd.read_csv("data/Diginetica/raw/yoochoose-clicks.dat")
    df = df[["session_id", "timestamp", "item_id"]]

    # filter sessions with length [10, 20]
    session_counts = df["session_id"].value_counts()
    sessions = []
    for i in range(MIN_NUM_NODES, MAX_NUM_NODES + 1, 1):
        valid_session_ids = session_counts[session_counts == i].index.tolist()
        assert len(valid_session_ids) >= NUM_EACH_LENGTH
        sessions.extend(random.sample(valid_session_ids, NUM_EACH_LENGTH))
    df = df[df["session_id"].isin(sessions)]

    # reindex item id
    print("Reindex item ids")
    old_item_ids = df["item_id"].drop_duplicates().tolist()
    item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_item_ids)}
    df["item_id"] = df["item_id"].apply(lambda x: item_id_mapping[x])

    # transform timestamp to Unix style
    print("Transform timestamp to Unix style")
    df["timestamp"] = df["timestamp"].apply(lambda x: rfc3339_to_unix_milliseconds(x))

    num_nodes = len(df["item_id"].drop_duplicates())
    print(f"Number of nodes: {num_nodes}")

    graphs = []
    for s in tqdm(
        range(NUM_SAMPLES),
        desc="Processing",
        unit="item",
    ):
        graph_data = {}
        selected_sessions = random.sample(sessions, NUM_SESSION_PER_SAMPLE)

        session_start_time = []
        for session_idx, session in enumerate(selected_sessions):
            session_df = df[df["session_id"] == session]
            session_df = session_df.sort_values("timestamp")
            session_start_time.append(session_df["timestamp"].min())
            session_item_ids = session_df["item_id"].tolist()
            session_srcs = session_item_ids[:-1]
            session_dsts = session_item_ids[1:]
            # for j in range(len(session_item_ids) - 1):
            #     session_srcs.append(j)
            #     session_dsts.append(j + 1)
            graph_data[("n", f"{session_idx}", "n")] = (
                torch.tensor(session_srcs),
                torch.tensor(session_dsts),
            )

        min_start_time = min(session_start_time)
        session_item_timestamp = []
        for start_time, session in zip(session_start_time, selected_sessions):
            time_offset = start_time - min_start_time
            session_df = df[df["session_id"] == session]
            session_df = session_df.sort_values("timestamp")
            session_item_timestamp.extend(
                [
                    (x["timestamp"] - time_offset, x["item_id"])
                    for _, x in session_df.iterrows()
                ]
            )
        sorted_session_item_timestamp = sorted(
            session_item_timestamp, key=lambda x: x[0]
        )
        _, item_ids = list(zip(*sorted_session_item_timestamp))
        mixed_srcs = item_ids[:-1]
        mixed_dsts = item_ids[1:]

        graph_data[("n", "origin", "n")] = (mixed_srcs, mixed_dsts)
        g = dgl.heterograph(graph_data, num_nodes_dict={"n": num_nodes})
        g = dgl.compact_graphs(g)

        graphs.append(g)
    dgl.save_graphs(
        os.path.join(PROCESSED_DIR, f"data_{NUM_SESSION_PER_SAMPLE}.bin"), graphs
    )


if __name__ == "__main__":
    seed_everything(12345)
    run()
