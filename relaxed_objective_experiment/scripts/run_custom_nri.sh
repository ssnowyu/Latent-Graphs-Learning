#!/bin/bash

python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.1/nri
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.3/nri
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.5/nri
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.7/nri
# python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.9/nri

python src/train.py -m seed=0,1,2,3,4 experiment=custom_10_0.3/nri
python src/train.py -m seed=0,1,2,3,4 experiment=custom_15_0.3/nri
python src/train.py -m seed=0,1,2,3,4 experiment=custom_20_0.3/nri
python src/train.py -m seed=0,1,2,3,4 experiment=custom_25_0.3/nri