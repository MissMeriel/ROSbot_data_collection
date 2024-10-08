#!/usr/bin/bash

. .venv/bin/activate

export PYTHONPATH=$(pwd):$(pwd)/../models:$PYTHONPATH
# echo $PYTHONPATH

# another test dataset ../datasets/rosbotxl_data
# test dataset in /p/rosbot/datasets1/rosbotxl_data/ is 1600 samples
# /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4 is about 11K samples
# datasetpath="/p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4"
datasetpath="/p/rosbot/datasets1/rosbotxl_data/ "
python3 train_DroNet.py $datasetpath --lossfxn MSE 