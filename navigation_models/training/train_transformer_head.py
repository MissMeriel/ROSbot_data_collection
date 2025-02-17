import sys, os
sys.path.append("../models/")

import shutil
from vit import vit_b_16, ViT_B_16_Weights
from test_backbone import *
import argparse

# from train_DAVE2
import numpy as np
import argparse
import pandas as pd
import matplotlib.image as mpimg
from torch.autograd import Variable
import matplotlib.pyplot as plt
from DatasetGenerator import MultiDirectoryDataSequence
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
import random, string
from torchvision.models.vision_transformer import *
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--backbone-id", type=str, default="")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--robustification", action='store_true')
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    return args

def train():
    start_time = time.time()
    input_shape = (224, 224) #(2560, 720)  # Example input shape: width x height
    backbone = ViT_B_16()
    head = LinearHead(in_features=768, out_features=1)
    backbone.eval()
    head.train()
    backbone.backbone.heads = head
    model = backbone
    args = parse_arguments()
    print(args)
    newpath = f"./transformer-training-output/transformer-head-{timestr()}-{randstr()}/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        shutil.copy(__file__, newpath+"/"+Path(__file__).name)
    # dataset = MultiDirectoryDataSequence(args.dataset, image_size=(input_shape[::-1]), transform=Compose([ToTensor()]),\
    #                                      robustification=args.robustification, noise_level=args.noisevar)
    datasets = []
    for d in args.dataset.split(","):
        dataset = MultiDirectoryDataSequence(d, image_size=(model.input_shape[::-1]), transform=model.preprocess,\
                                         robustification=args.robustification, noise_level=args.noisevar)
        print(f"{dataset.get_total_samples()=}, {len(dataset)}")
        print(f"Moments of {d}  distribution:", dataset.get_outputs_distribution())
        datasets.append(dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    # print("Retrieving output distribution....")
    # print("Moments of distribution:", dataset.get_outputs_distribution())
    # print("Total samples:", dataset.get_total_samples(), flush=True)
    print("Individual samples:", dataset.cumulative_sizes, "\nTotal samples:", sum(dataset.cumulative_sizes), flush=True)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    print("time to load dataset: {}".format(time.time() - start_time))

    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-{args.epochs}epoch-{int(sum(dataset.cumulative_sizes)/1000)}Ksamples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"Running on {device=}", flush=True)
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20
    for epoch in range(args.epochs):
        epoch_start_time = time.time() # Record start time for the epoch
        running_loss = 0.0
        sampled_loss = np.zeros(10)
        for i, hashmap in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            x = hashmap['image name'].float().to(device)
            y = hashmap['angular_speed_z'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            # forward + backward + optimize
            outputs = model(x)
            # loss = F.mse_loss(outputs.flatten(), y)
            loss = F.mse_loss(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print(f"{np.var(sampled_loss)}    {sampled_loss=}")
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq))
                sampled_loss = np.roll(sampled_loss, 1)
                sampled_loss[0] = running_loss / logfreq
                if (running_loss / logfreq) < lowest_loss:
                    print(f"New best model! MSE loss: {running_loss / logfreq}", flush=True)
                    model_name = f"./{newpath}/model-{iteration}-best.pt"
                    print(f"Saving model to {model_name}")
                    torch.save(model, model_name)
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0

        epoch_end_time = time.time()  # Record end time for the epoch
        epoch_duration = epoch_end_time - epoch_start_time # Record total time for the epoch
        print("Epoch Train Duration: {}".format(epoch_duration))
        print(f"Finished {epoch=}")
        sys.stdout.flush()
        model_name = f"./{newpath}/model-{iteration}-epoch{epoch}.pt"
        print(f"Saving model to {model_name}")
        torch.save(model, model_name)
        if np.var(sampled_loss) < 0.005:
            print(f"Loss converged at {loss} (variance {np.var(sampled_loss):.4f}); quitting training...")
            break
    print('Finished Training')

    # save state dict
    model_name = f"./{newpath}/model-statedict-{iteration}-epoch{epoch}.pt"
    torch.save(model.state_dict(), model_name)

    # delete models from previous epochs
    # print("Deleting models from previous epochs...")
    # for epoch in range(args.epochs):
    #     os.remove(f"./{newpath}/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    with open(f'./{newpath}/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={int(sum(dataset.cumulative_sizes))}\n"
                f"{args.epochs=}\n"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{args.robustification=}\n"
                f"{args.noisevar=}\n"
                # f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")

def test():
    backbone = ViT_B_16()
    head = LinearHead(in_features=768, out_features=1)
    backbone.backbone.heads = head
    # print(backbone)
    out = backbone(torch.rand((1, 3, 224, 224)))
    pytorch_model_size(backbone)
    print(f"{out.shape=}, {out.item()=}")

def test2():
    backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    pytorch_model_size(backbone)
    print(backbone.image_size)
    print(backbone.heads)
    head = nn.Linear(in_features=768, out_features=1, bias=True)
    backbone.heads = nn.Sequential(head, nn.Hardtanh(-1, 1))
    # print(backbone)
    pytorch_model_size(backbone)
    for i in range(10):
        out = backbone(torch.rand((1, 3, backbone.image_size, backbone.image_size)))
        print(f"{out.shape=}, {out.item()=}")

if __name__ == '__main__':
    # train()
    test2()