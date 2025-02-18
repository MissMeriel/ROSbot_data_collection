#!/usr/bin/bash

#SBATCH --mem=22G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ms7nk@virginia.edu

. ../../.venv-rb/bin/activate
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
# test dataset in /p/rosbot/datasets1/rosbotxl_data/ is 1600 samples
# /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4 is about 11K samples
dataset_directory="../../dataset/data-meriel,../../dataset/data-zach,../../dataset/data-math-and-johann,../../dataset/data-yili"
python3 train_transformer_head2.py $dataset_directory  --batch 16 --robustification --slurmid $SLURM_JOB_ID