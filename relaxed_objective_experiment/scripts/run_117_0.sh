#!/bin/bash

python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.2/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.4/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.6/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.8/gat data.batch_size=16

python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.1/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.3/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.5/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.7/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.9/gat data.batch_size=16
