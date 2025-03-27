#!/bin/bash

python train.py -m seed=0,1,2,3,4 experiment=stac_15/sage data.batch_size=4
# python train.py -m seed=0,1,2,3,4 experiment=stac_15/gcn data.batch_size=8
python train.py -m seed=0,1,2,3,4 experiment=stac_15/gat data.batch_size=4
# python train.py -m seed=0,1,2,3,4 experiment=stac_15/nri data.batch_size=4
