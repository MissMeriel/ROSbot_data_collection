#!/usr/bin/bash


#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ms7nk@virginia.edu
. ../.venv-rb/bin/activate
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
# test dataset in /p/rosbot/datasets1/rosbotxl_data/ is 1600 samples
# /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4 is about 11K samples
python3 train_transformer_head2.py /p/rosbot/rosbotxl/data-yili/cleaned/mecanum_wheels/rosbotxl_off_4  --batch 16 --robustification