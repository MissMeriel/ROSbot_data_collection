#!/usr/bin/bash
. .venv-rb/bin/activate
arg="$1"
echo Downloading $arg
# download failure catalog to failure-catalog
# if [[ "$arg" = "FAILURES" || "$arg" = "ALL"  ]]; then
#     gdown --folder  https://drive.google.com/drive/folders/1Lntd0lctZ05JxOc6pdGFCkYDMpMbI_cN
#     for file in failure-catalog/*; do
#         if (file $file | grep -q compression ) ; then
#             echo "found $file, unzipping to ${file%.*}"
#             unzip $file -d failure-catalog/
#         fi
#     done
# fi
# # download training dataset to dataset
# if [[ "$arg" = "DATASET" || "$arg" = "ALL"  ]]; then
#     gdown --folder  https://drive.google.com/drive/folders/1Zn7ZNDpPpw7ffnotwR8Jb-DGITRNZy0A
#     for file in dataset/*; do
#         if (file $file | grep -q compression ) ; then
#             echo "found $file, unzipping to ${file%.*}"
#             unzip $file -d dataset/
#         fi
#     done
# fi
# # download pretrained models to pretrained-models
# if [[ "$arg" = "MODELS" || "$arg" = "ALL"  ]]; then
#     gdown --folder  https://drive.google.com/drive/folders/1lTqEC30yBuqN6IobSV73E97OaOnCGqDg
#     for file in pretrained-models/*; do
#         if (file $file | grep -q compression ) ; then
#             echo "found $file, unzipping to ${file%.*}"
#             unzip $file -d pretrained-models/
#         fi
#     done
# fi