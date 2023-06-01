#!/usr/bin/bash

python -m venv .venv
. .venv/bin/activate

python train_DAVE2.py ../datasets/<dataset-parentdir>