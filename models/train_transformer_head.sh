#!/usr/bin bash
. ../../training/.venv/bin/activate
# test dataset in /p/rosbot/datasets1/rosbotxl_data/ is 1600 samples
python3 train_transformer_head.py /p/rosbot/datasets1/rosbotxl_data/