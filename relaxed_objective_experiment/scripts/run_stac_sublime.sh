#!/bin/bash

# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_5/sublime data.batch_size=64
# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_10/sublime data.batch_size=32
# python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_15/sublime data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_20/sublime data.batch_size=8
python src/train.py -m seed=0,1,2,3,4 experiment=labeled_stac_25/sublime data.batch_size=4