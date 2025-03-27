#!/bin/bash

# python src/train.py -m seed=1,2,3,4 experiment=labeled_stac_5/gaug data.batch_size=64
# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_10/gaug data.batch_size=32
# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_15/gaug data.batch_size=16
# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_20/gaug data.batch_size=8
# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_25/gaug data.batch_size=4

python src/train.py -m seed=0,1 experiment=custom_10_0.3/sage data.batch_size=2
python src/train.py -m seed=0,1,2,3,4 experiment=custom_15_0.3/sage data.batch_size=2
python src/train.py -m seed=0,1,2,3,4 experiment=custom_20_0.3/sage data.batch_size=2
python src/train.py -m seed=0,1,2,3,4 experiment=custom_25_0.3/sage data.batch_size=1