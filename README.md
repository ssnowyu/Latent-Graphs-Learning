# 1 Introduction
The file structure is as follows:
```

├── data_simulation_tool/          # Code for data simulation tool
├── lgl_experiment/               # Code for LGL task main experiment
├── relaxed_objective_experiment/  # Code for relaxed objective experiment
```

# 2 Data Simulation Tool
## 2.1 Quickstart
The process of generating data using the data tool is as follows:
1. Access path `data_simulation_tool`
```bash
cd data_simulation_tool
```
2. Create an environment
```Bash
conda create -n data_sim python=3.9

conda activate data_sim

pip install -r requirements.txt
```
3. Generate data
```bash
cd src

python to_csv.py
```

Successful execution will result in three csv files, and the processing code to convert these csv files into dataset is provided in `lgl_experiment/src/data`.
```
chains.csv  # groud truth for GSL with original objectie
graph.csv  # groud truth for GSL with relaxed objectie
snapshots.csv  # input data
```

All parameters can be modified in the configuration file `data_simulation_tool/configs/generator/original_condition.yaml`

# 3 Reproducing the experiment
## 3.1 LGL main experiment

### 3.1.1 Quickstart
Please reproduce the results of the experiment by following the steps below:
1. Access path `lgl_experiment`
```bash
cd lgl_experiment
```
2. Create an environment
```Bash
conda create -n exp_env1 python=3.9

conda activate exp_env1

pip install -r requirements.txt
```

3. Download data from [data](https://drive.google.com/file/d/1f3uxQQYwag36ID4h6ohFuvBnk7qZAdPr/view?usp=sharing) and unzip it to `lgl_experiment/data`.

4. We used ClearML as a presentation tool for the results. Therefore a connection to ClearML is required. please refer to https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps

5. Train models
```bash
cd src

python train.py experiment=gcn_for_chain_based_on_local_data_with_0.1_noises
```
The value of `experiment` can be replaced with the names of all files in the `lgl_experiment/configs/experiment` path that are suffixed with `0.1_noises`.

## 3.2 Reproducing the relaxed objective experiment

### 3.2.1 Quickstart
1. Access path `relaxed_objective_experiment`

```bash
cd relaxed_objective_experiment
```

2. Create an environment

```Bash
conda create -n exp_env2 python=3.9

conda activate exp_env2

pip install -r requirements.txt
```

3. Download data from [data](https://drive.google.com/file/d/17RC3fKrOKGqYV3lA--hq_zKmJjLCmq0J/view?usp=sharing) and unzip it to `relaxed_objective_experiment/data`

4. (Optional) Generate synthetic data, or use our pre-generated synthetic data in `relaxed_objective_experiment/data/custom/`

```bash
# generate synthetic data
cd src/data_process

python to_csv.py
```

Successful execution will result in a **.bin** file in `data/custom/processed`.

5. Train models

```bash
cd src

python train.py experiment=custom_5_0.3/sage
```

The `0.3` above can be replaced by `0.1`, `0.2`, `0.4`, `0.5`, `0.6`, `0.7`, `0.8`.
