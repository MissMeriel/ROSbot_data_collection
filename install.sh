#!/bin/bash
# install python environment
module load python-3.10.13
python3.10 venv .venv-rb
. .venv-rb/bin/activate
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install -r requirements.txt
# download catalog
./download.sh ALL
