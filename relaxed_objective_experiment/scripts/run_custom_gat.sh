#!/bin/bash

python src/train.py -m seed=3,4 experiment=custom_5_0.2/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.4/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.6/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.8/gat data.batch_size=16

python src/train.py -m seed=3,4 experiment=custom_5_0.1/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.3/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.5/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.7/gat data.batch_size=16
python src/train.py -m seed=3,4 experiment=custom_5_0.9/gat data.batch_size=16

python src/train.py -m seed=3,4 experiment=custom_10_0.3/gat data.batch_size=8
python src/train.py -m seed=3,4 experiment=custom_15_0.3/gat data.batch_size=8
python src/train.py -m seed=3,4 experiment=custom_20_0.3/gat data.batch_size=4
python src/train.py -m seed=3,4 experiment=custom_25_0.3/gat data.batch_size=2
