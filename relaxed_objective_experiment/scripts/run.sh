#!/bin/bash

cd ../src
nohup python train.py -m seed=0,1,2,3,4 experiment=diginetica_10/gcn >./out/0001.out &
nohup python train.py -m seed=0,1,2,3,4 experiment=diginetica_10/sage >./out/0002.out &
nohup python train.py -m seed=0,1,2,3,4 experiment=diginetica_10/gat >./out/0003.out &
# nohup python train.py -m seed=0,1,2,3,4 experiment=diginetica_10/nri >./out/0004.out &