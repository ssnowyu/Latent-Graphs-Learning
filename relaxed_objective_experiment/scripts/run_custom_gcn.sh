#!/bin/bash

python src/train.py -m seed=2 experiment=custom_5_0.2/gcn data.batch_size=8
python src/train.py -m seed=2 experiment=custom_5_0.4/gcn data.batch_size=8
python src/train.py -m seed=2 experiment=custom_5_0.6/gcn data.batch_size=8
python src/train.py -m seed=2 experiment=custom_5_0.8/gcn data.batch_size=8

# python src/train.py -m seed=2 experiment=custom_5_0.1/gcn data.batch_size=8
# python src/train.py -m seed=2 experiment=custom_5_0.3/gcn data.batch_size=8
# python src/train.py -m seed=2 experiment=custom_5_0.5/gcn data.batch_size=8
# python src/train.py -m seed=2 experiment=custom_5_0.7/gcn data.batch_size=8
# python src/train.py -m seed=2 experiment=custom_5_0.9/gcn data.batch_size=8

python src/train.py -m seed=2 experiment=custom_10_0.3/gcn data.batch_size=4
python src/train.py -m seed=2 experiment=custom_15_0.3/gcn data.batch_size=4
python src/train.py -m seed=2 experiment=custom_20_0.3/gcn data.batch_size=2
python src/train.py -m seed=2 experiment=custom_25_0.3/gcn data.batch_size=2