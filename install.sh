#!/usr/bin/bash
module load python-3.10.13
python3.10 venv .venv-rb
. .venv-rb/bin/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
