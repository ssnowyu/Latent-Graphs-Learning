#!/bin/bash

python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.2/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.4/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.6/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.8/sage data.batch_size=16

python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.1/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.3/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.5/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.7/sage data.batch_size=16
python src/train.py -m seed=0,1,2,3,4 experiment=custom_5_0.9/sage data.batch_size=16
