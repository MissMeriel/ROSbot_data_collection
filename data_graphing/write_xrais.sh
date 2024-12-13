#!/usr/bin/bash


#SBATCH --mem-per-cpu=28Gb

. ../.venv-rb/bin/activate

python3 write_all_XRAIs.py --passingtrace /p/rosbot/rosbotxl/deployment-yili/M1/F3/