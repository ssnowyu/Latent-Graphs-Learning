import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import os
import json
import dgl
import random
import numpy as np
from transformers import AutoTokenizer, BertModel
import torch
from tqdm import tqdm

from src.utils.seed_utils import seed_everything
from src.utils.graph_utils import complete_graph


MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RAW_DIR = "data/STAC/raw"
PROCESSED_DIR = "data/STAC/processed"
DIM_FEATURE = 768
NUM_DIALOGUES_PER_SAMPLE = 25
NUM_SAMPLES = 2000


def merge_raw_dialogues():
    raw_dir = "data/STAC/raw/raw"
    processed_dir = "data/STAC/raw"

    train_data_path = os.path.join(raw_dir, "train.json")
    val_data_path = os.path.join(raw_dir, "val.json")
    test_data_path = os.path.join(raw_dir, "test.json")

    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_data_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    data = train_data + val_data + test_data

    # remove empty data
    data = [
        item
        for item in data
        if (len(item["relations"]) != 0 and len(item["edus"]) != 0)
    ]

    with open(
        os.path.join(processed_dir, "dialogues.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(data, f)


def generate_time(local: float, scale: float, size=None):
    time = -1.0
    while time <= 0:
        time = np.random.normal(local, scale, size)
    return time


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device=DEVICE)
    model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)

    with open(os.path.join(RAW_DIR, "dialogues.json"), "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    random.shuffle(dialogues)

    graphs = []
    for _ in tqdm(
        range(NUM_SAMPLES),
        desc="Processing",
        unit="item",
    ):
        # sample_dialogues = dialogues[
        #     i * NUM_DIALOGUES_PER_SAMPLE : (i + 1) * NUM_DIALOGUES_PER_SAMPLE
        # ]

        sample_dialogues = random.sample(dialogues, NUM_DIALOGUES_PER_SAMPLE)

        edu_nums = [len(item["edus"]) for item in sample_dialogues]
        num_all_edus = sum(edu_nums)

        texts = []
        edges = []
        node_index_offset = 0
        for dialogue in sample_dialogues:
            dialogue_texts = [item["text"] for item in dialogue["edus"]]
            dialogue_edges = [
                (item["x"] + node_index_offset, item["y"] + node_index_offset)
                for item in dialogue["relations"]
            ]
            texts.extend(dialogue_texts)
            edges.append(dialogue_edges)

            node_index_offset += len(dialogue["edus"])

        # text embeddings for edus
        text_embeddings = []
        for text in texts:
            encoding = tokenizer(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(DEVICE)
            outputs = model(**encoding)
            text_embeddings.append(outputs.pooler_output.detach())
        text_embeddings = torch.cat(text_embeddings, dim=0).cpu()
        # text_embeddings = torch.zeros((len(texts), 768))

        # merge all dialogues
        edu_timestamps = []
        for num in edu_nums:
            start_time = np.random.uniform(0, np.mean(edu_nums))
            timestamps = [start_time]
            for i in range(num - 1):
                timestamps.append(timestamps[i] + generate_time(1, 1))
            edu_timestamps.extend(timestamps)

        sorted_indexes = np.argsort(edu_timestamps).tolist()
        new_index_mapper = {
            old_id: new_id for new_id, old_id in enumerate(sorted_indexes)
        }
        # error_edges = [
        #     [(sorted_indexes[item[0]], sorted_indexes[item[1]]) for item in each_edges]
        #     for each_edges in edges
        # ]
        edges = [
            [
                (new_index_mapper[item[0]], new_index_mapper[item[1]])
                for item in each_edges
            ]
            for each_edges in edges
        ]
        num_nodes = num_all_edus

        # add edges to semi-complete graph
        complete_g = complete_graph(num_nodes)
        added_srcs, added_dsts = complete_g.edges()

        mask = added_srcs <= added_dsts
        added_srcs = added_srcs[mask]
        added_dsts = added_dsts[mask]

        # srcs = torch.cat((torch.tensor(list(range(num_nodes - 1))), added_srcs))
        # dsts = torch.cat((torch.tensor(list(range(1, num_nodes, 1))), added_dsts))

        # added_edges = torch.stack((srcs, dsts), dim=0).unique(sorted=False, dim=1)

        # srcs = added_edges[0, :]
        # dsts = added_edges[1, :]

        # create graph
        graph_data = {
            ("n", "origin", "n"): (added_srcs, added_dsts),
        }

        for idx, dialogue_edges in enumerate(edges):
            srcs, dsts = list(zip(*dialogue_edges))
            graph_data[("n", f"{idx}", "n")] = (torch.tensor(srcs), torch.tensor(dsts))

        g = dgl.heterograph(graph_data, num_nodes_dict={"n": num_nodes})
        g.ndata["feat"] = text_embeddings

        graphs.append(g)
    dgl.save_graphs(
        os.path.join(PROCESSED_DIR, f"data_{NUM_DIALOGUES_PER_SAMPLE}.bin"), graphs
    )


if __name__ == "__main__":
    seed_everything(12345)
    # merge_raw_dialogues()

    run()
