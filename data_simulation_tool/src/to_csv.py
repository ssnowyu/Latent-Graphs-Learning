import os.path

import pyrootutils
import py7zr

# set project root dir to PYTHONPATH
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import datetime
import time
import pandas as pd
from src.data_generator import DataGenerator
import yaml


def decompress_data():
    raw_file_dir = '../data/raw'
    archive = py7zr.SevenZipFile(os.path.join(raw_file_dir, 'call_graph.7z'), mode="r")
    archive.extractall(path=raw_file_dir)
    archive.close()


if __name__ == "__main__":
    raw_data_file = '../data/raw/call_graph.csv'
    if not os.path.exists(raw_data_file):
        print("decompress data")
        decompress_data()

    filepath = "../configs/generator/original_condition.yaml"
    with open(filepath, "r", encoding="utf-8") as f:
        params_str = f.read()
        print(f"load config from {filepath}")
        print("=====================================================")
        print(f"{params_str}\n")
        params = yaml.load(params_str, Loader=yaml.FullLoader)

    # generator = DataGenerator(**params)
    generator = DataGenerator(
        data_path=params["data_path"],
        num_chain=params["num_chain"],
        num_step=params["num_step"],
        interval_time=params["interval_time"],
        observed_time=params["observed_time"],
        chain_types=params["chain_types"],
        chain_meta_distribution_params=params["chain_meta_distribution_params"],
        chain_type_ratio=params["chain_type_ratio"],
        edge_types=params["edge_types"],
        edge_meta_distribution_params=params["edge_meta_distribution_params"],
        edge_type_ratio=params["edge_type_ratio"],
        start_steps=params["start_steps"],
        noise_ratio=params["noise_ratio"],
        min_chain_len=params["min_chain_len"],
        max_chain_len=params["max_chain_len"],
        min_execute_time=params["min_execute_time"],
        max_execute_time=params["max_execute_time"],
        distribution_params_corrected=params["distribution_params_corrected"],
        execute_edge_chain_ratio=params["execute_edge_chain_ratio"],
    )
    all_chain_df = []
    all_snapshot_df = []
    all_global_graph_df = []

    # num_samples = 1000
    for i in range(params["num_samples"]):
        print(f"\ngenerating the {i}-th sample")
        start = time.perf_counter()
        generator.generate_data()
        chain_df, snapshot_df, global_graph_df = generator.to_dataframe()
        all_chain_df.append(chain_df)
        all_snapshot_df.append(snapshot_df)
        all_global_graph_df.append(global_graph_df)
        end = time.perf_counter()
        print(f"execute time is {end - start}")

    print("merging results")
    all_chain_df = pd.concat(all_chain_df, axis=0)
    all_snapshot_df = pd.concat(all_snapshot_df, axis=0)
    all_global_graph_df = pd.concat(all_global_graph_df, axis=0)

    current_time = datetime.datetime.now()
    time_format = "%Y-%m-%d-%H-%M-%S"
    time_str = current_time.strftime(time_format)

    print("saving")
    all_chain_df.to_csv(f"../data/result/{time_str}-chains.csv", index=False)
    all_snapshot_df.to_csv(f"../data/result/{time_str}-snapshots.csv", index=False)
    all_global_graph_df.to_csv(f"../data/result/{time_str}-graph.csv", index=False)

    print("complete")
