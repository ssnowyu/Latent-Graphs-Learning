#!/bin/bash

# python src/train.py -m seed=0,1,2,3,4 experiment=stac_5/gcn data.batch_size=64
# python src/train.py -m seed=0,1,2,3,4 experiment=stac_10/gcn data.batch_size=32
# python src/train.py -m seed=0,1,2,3,4 experiment=stac_15/gcn data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=stac_20/gcn data.batch_size=4
python src/train.py -m seed=0,1,2,3,4 experiment=stac_25/gcn data.batch_size=2