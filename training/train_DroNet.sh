#!/usr/bin/bash

. .venv/bin/activate

export PYTHONPATH=$(pwd):$(pwd)/../models:$PYTHONPATH
echo $PYTHONPATH

python3 train_DroNet.py # --dataset rosbot --epochs 100