#!/bin/bash

# python src/train.py -m seed=0,1,2,3,4 experiment=diginetica_5/gat data.batch_size=64
# python src/train.py -m seed=0,1,2,3,4 experiment=diginetica_10/gat data.batch_size=32
# python src/train.py -m seed=0,1,2,3,4 experiment=diginetica_15/gat data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=diginetica_20/gat data.batch_size=1
python src/train.py -m seed=0,1,2,3,4 experiment=diginetica_25/gat data.batch_size=1
python src/train.py -m seed=0,1,2,3,4 experiment=stac_20/sage data.batch_size=1
python src/train.py -m seed=0,1,2,3,4 experiment=stac_25/sage data.batch_size=1