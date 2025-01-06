#!/usr/bin/bash
module load python-3.10.13
python3.10 venv .venv-rb
. .venv-rb/bin/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
# download failure catalog to failure-catalog
gdown --folder  https://drive.google.com/drive/folders/1Lntd0lctZ05JxOc6pdGFCkYDMpMbI_cN
for file in failure-catalog/*; do
    if (file $file | grep -q compression ) ; then
        echo "found $file, unzipping to ${file%.*}"
        unzip $file -d failure-catalog/
    fi
done
# download training dataset to dataset
gdown --folder  https://drive.google.com/drive/folders/1Zn7ZNDpPpw7ffnotwR8Jb-DGITRNZy0A
for file in dataset/*; do
    if (file $file | grep -q compression ) ; then
        echo "found $file, unzipping to ${file%.*}"
        unzip $file -d dataset/
    fi
done
# download pretrained models to pretrained-models
gdown --folder  https://drive.google.com/drive/folders/1lTqEC30yBuqN6IobSV73E97OaOnCGqDg
for file in pretrained-models/*; do
    if (file $file | grep -q compression ) ; then
        echo "found $file, unzipping to ${file%.*}"
        unzip $file -d pretrained-models/
    fi
done