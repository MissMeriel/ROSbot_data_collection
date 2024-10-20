#!/usr/bin/bash

#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ms7nk@virginia.edu

. .venv/bin/activate
export PYTHONPATH=$(pwd):$(pwd)/../models:$PYTHONPATH
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# another test dataset ../datasets/rosbotxl_data
# math & johann: /p/rosbot/rosbotxl/data-math-and-johann/cleaned/mecanum_wheels/south_hall_rice_4floor/
# math & johann: /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/donuts_rice_4floor/
# math & johann: /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/loops_rice_4floor
# math & johann: /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/loopsv2_rice_4floor
# math & johann: /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/south_hall_rice_4floor
# zach: /p/rosbot/rosbotxl/zach_data/data_collections/organized/not_tilted/
# zach: /p/rosbot/rosbotxl/zach_data/data_collections/organized/tilted/
# zach: /p/rosbot/rosbotxl/zach_data/data_collections/organized/rosbotxl_data/
# test dataset in /p/rosbot/datasets1/rosbotxl_data/ is 1600 samples
# /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4 is about 11K samples
# python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-math-and-johann/cleaned/mecanum_wheels/south_hall_rice_4floor/  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/donuts_rice_4floor/  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/loops_rice_4floor/  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/loopsv2_rice_4floor  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/data-math-and-johann/cleaned/regular_wheels/south_hall_rice_4floor  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/zach_data/data_collections/organized/not_tilted/ --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/zach_data/data_collections/organized/tilted/  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/rosbotxl/zach_data/data_collections/organized/rosbotxl_data/  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
python3 train_DAVE2.py  /p/rosbot/datasets1/rosbotxl_data/156  --batch 32 --lossfxn MSE --archid DAVE2v3 --robustification --convergence --slurmid $SLURM_JOB_ID
