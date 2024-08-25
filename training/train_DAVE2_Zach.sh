#!/usr/bin/bash

. .venv/bin/activate

export PYTHONPATH=$(pwd):$(pwd)/../models:$PYTHONPATH

python3 training/Zach_train_DAVE2.py datasets/rosbotxl_data
