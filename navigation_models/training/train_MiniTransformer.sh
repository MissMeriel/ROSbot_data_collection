#!/usr/bin/bash

. .venv/bin/activate

export PYTHONPATH=$(pwd):$(pwd)/../models:$PYTHONPATH
echo $PYTHONPATH

python3 train_MiniTransformer.py --dataset rosbot --epochs 100
# python train_MiniTransformer.py --dataset fmnist
# python train_MiniTransformer.py --dataset svhn --n_channels 3 --image_size 32 --embed_dim 128
# python train_MiniTransformer.py --dataset cifar10 --n_channels 3 --image_size 32 --embed_dim 128
