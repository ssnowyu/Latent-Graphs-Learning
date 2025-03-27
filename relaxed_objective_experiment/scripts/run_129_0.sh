#!/bin/bash

python src/train.py -m seed=0,1,2,3,4 experiment=custom_10_0.3/sage data.batch_size=4
python src/train.py -m seed=0,1,2,3,4 experiment=custom_15_0.3/sage data.batch_size=4
python src/train.py -m seed=0,1,2,3,4 experiment=custom_20_0.3/sage data.batch_size=2
python src/train.py -m seed=0,1,2,3,4 experiment=custom_25_0.3/sage data.batch_size=2