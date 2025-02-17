import sys, os
sys.path.append("../models")
import argparse
import shutil
# from train_DAVE2
import numpy as np
import argparse
# import pandas as pd
# import matplotlib.image as mpimg
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
import random #, string
from torchvision.models.vision_transformer import *
from utils import *
from pathlib import Path
import shutil

# other ViTs check:
# https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--backbone-id", type=str, default="")
    parser.add_argument("--lossfxn", type=str, default="MSE")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--robustification", action='store_true')
    parser.add_argument("--convergence", action='store_true')
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--slurmid", type=int, default=0)
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    return args

def train():
    start_time = time.time()
    torch.cuda.empty_cache()
    # backbone
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    pytorch_model_size(model)
    print(model.image_size, flush=True)
    # head
    print(model.heads, flush=True)
    pytorch_model_size(model)
    model.eval()
    head = nn.Linear(in_features=768, out_features=1, bias=True)
    head.train()
    model.heads = nn.Sequential(head, nn.Hardtanh(-1, 1))
    input_shape = (model.image_size, model.image_size) #(2560, 720)  # Example input shape: width x height
    # backbone.backbone.heads = head
    # model = backbone
    args = parse_arguments()
    print(args, flush=True)
    newpath = f"./transformer-training-output/transformer-head-{timestr()}-{randstr()}/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        shutil.copy(__file__, newpath+"/"+Path(__file__).name)
    if args.normalize:
        transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = Compose([ToTensor()])
    datasets = []
    for d in args.dataset.split(","):
        dataset = MultiDirectoryDataSequence(d, image_size=input_shape, transform=transform,\
                                         robustification=args.robustification, noise_level=args.noisevar)
        print(f"{dataset.get_total_samples()=}, {len(dataset)}")
        print(f"Moments of {d}  distribution:", dataset.get_outputs_distribution())
        datasets.append(dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    print("Individual samples:", dataset.cumulative_sizes, "\nTotal samples:", dataset.cumulative_sizes[-1], flush=True)
    all_outputs = np.array([])
    for ds in dataset.datasets:
        for key in ds.dfs_hashmap.keys():
            df = ds.dfs_hashmap[key]
            arr = df['angular_speed_z'].to_numpy()
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
    dataset_moments = get_outputs_distribution(all_outputs)
    print(f"{dataset_moments=}")

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    print("time to load dataset: {}".format(time.time() - start_time))

    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-loss{args.lossfxn}-aug{args.robustification}-converge{args.convergence}-norm{args.normalize}-{args.epochs}epoch-{args.batch}batch-{int(dataset.cumulative_sizes[-1]/1000)}Ksamples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"Training on {device=}", flush=True)
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20
    if args.lossfxn == "L1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    sampled_loss = np.ones(10)
    for epoch in range(args.epochs):
        epoch_start_time = time.time() # Record start time for the epoch
        running_loss = 0.0
        for i, hashmap in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            x = hashmap['image_name'].float().to(device)
            y = hashmap['angular_speed_z'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            # forward + backward + optimize
            outputs = model(x)
            # loss = F.mse_loss(outputs.flatten(), y)
            loss = criterion(outputs, y)
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
                    model_name = f"{newpath}/model-{iteration}-best.pt"
                    print(f"New best model! MSE loss: {running_loss / logfreq}\nSaving model to {model_name}", flush=True)
                    torch.save(model, model_name)
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0

        epoch_end_time = time.time()  # Record end time for the epoch
        epoch_duration = epoch_end_time - epoch_start_time # Record total time for the epoch
        model_name = f"{newpath}/model-{iteration}-epoch{epoch}.pt"
        print(f"Finished Epoch {epoch+1}\nEpoch Duration: {epoch_duration:.1f}s\nSaving model to {model_name}", flush=True)
        sys.stdout.flush()
        torch.save(model, model_name)
        print(f"{np.var(sampled_loss)=}", flush=True)
        if args.convergence and np.var(sampled_loss) < 0.0001:
            print(f"Loss converged at {loss} (variance {np.var(sampled_loss):.4f}); quitting training...", flush=True)
            break
    print('Finished Training')

    # save state dict
    model_name = f"./{newpath}/model-statedict-{iteration}-epoch{epoch}.pt"
    torch.save(model.state_dict(), model_name)

    # delete models from previous epochs
    print("Deleting models from previous epochs...")
    for epoch in range(args.epochs):
        os.remove(f"{newpath}/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    with open(f'./{newpath}/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.cumulative_sizes[-1]}\n"
                f"{args.epochs=}\n"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{args.robustification=}\n"
                f"{args.noisevar=}\n"
                f"dataset_moments={dataset_moments}\n"
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
    train()
