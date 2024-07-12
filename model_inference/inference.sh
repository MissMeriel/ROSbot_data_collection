#!/usr/bin/bash
#SBATCH --mem-per-cpu=100G

python3 model_inference.py --dataset_dir /u/afr8gr/ROSbot_data_collection/datasets/TestData3 --models_dir /u/afr8gr/ROSbot_data_collection/models/Dave2-Keras-off-4-plus-corner-full
