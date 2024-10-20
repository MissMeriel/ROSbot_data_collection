#!/usr/bin/bash

#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ms7nk@virginia.edu

. .venv/bin/activate
export PYTHONPATH=$(pwd):$(pwd)/../models:$PYTHONPATH
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# another test dataset ../datasets/rosbotxl_data
# test dataset in /p/rosbot/datasets1/rosbotxl_data/ is 1600 samples
# /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4 is about 11K samples
python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4 --batch 32 --lossfxn MSE --archid DAVE2v1 --convergence --slurmid $SLURM_JOB_ID