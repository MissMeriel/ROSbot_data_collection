import numpy as np
import argparse
import pandas as pd
import matplotlib.image as mpimg
from torch.autograd import Variable
import shutil
from pathlib import Path
import os
import matplotlib.pyplot as plt
from DatasetGenerator import MultiDirectoryDataSequence
import time
import sys
sys.path.append("../models")
from ShuffleNetHead import *
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
from utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--robustification", action='store_true')
    parser.add_argument("--convergence", action='store_true')
    # parser.add_argument("--normalize", action='store_true') # mobilenet is always normalized, see MobileNetHead.preprocess
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--slurmid", type=int, default=0)
    # parser.add_argument("--archid", type=str, default="MobileNetHead")
    parser.add_argument("--lossfxn", type=str, default="MSE")
    parser.add_argument("--resize", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print(args)
    start_time = time.time()
    if args.resize is not None:
        dims = [int(i) for i in args.resize.split("x")]
        input_shape = tuple(dims)
    else:
        input_shape = (2560, 720)  # Example input shape: width x height
    model = ShuffleNetSteer(input_shape)
    datasets = []
    for d in args.dataset.split(","):
        dataset = MultiDirectoryDataSequence(d, image_size=(model.input_shape[::-1]), transform=model.preprocess,\
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
    print(f"Time to load dataset: {(time.time() - start_time):.1f}s", flush=True)
    outdir = f"./mobilenet-training-output-TESTMOMENTS/{model._get_name()}-{randstr()}-{timestr()}-{args.slurmid}/"
    os.makedirs(outdir, exist_ok=True)
    shutil.copy(__file__, outdir+"/"+Path(__file__).name)
    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-loss{args.lossfxn}-aug{args.robustification}-converge{args.convergence}-{args.epochs}epoch-{args.batch}batch-{int(dataset.cumulative_sizes[-1]/1000)}Ksamples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model.train()
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = args.log_interval
    if args.lossfxn == "L1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    sampled_loss = np.ones(10)
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time() # Record start time for the epoch
        running_loss = 0.0
        epoch_loss = 0.0
        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image_name'].float().to(device)
            y = hashmap['angular_speed_z'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / logfreq))
                # sampled_loss = np.roll(sampled_loss, 1)
                # sampled_loss[0] = running_loss / logfreq
                if epoch > 9 and abs(running_loss / logfreq) < lowest_loss:
                    print(f"New best model! {args.lossfxn} Loss: {running_loss / logfreq}", flush=True)
                    model_name = f"{outdir}/model-{iteration}-best.pt"
                    print(f"Saving model to {model_name}", flush=True)
                    torch.save(model, model_name)
                    lowest_loss = abs(running_loss / logfreq)
                running_loss = 0.0
        epoch_end_time = time.time()  # Record end time for the epoch
        epoch_duration = epoch_end_time - epoch_start_time # Record total time for the epoch
        sampled_loss = np.roll(sampled_loss, 1)
        sampled_loss[0] = epoch_loss
        model_name = f"{outdir}/model-{iteration}-epoch{epoch}.pt"
        print(f"Finished Epoch {epoch + 1}\nEpoch Duration: {epoch_duration:.1f}s\nEpoch Loss: {epoch_loss:.5f}\nSaving model to {model_name}", flush=True)
        sys.stdout.flush()
        torch.save(model, model_name)
        print(f"{np.var(sampled_loss)=}", flush=True)
        if args.convergence and np.var(sampled_loss) < 0.001:
            print(f"Loss converged at {loss} (variance {np.var(sampled_loss):.4f}); quitting training...", flush=True)
            break
    print('Finished Training')

    # save model
    model_name = f'{outdir}/model-{iteration}.pt'
    torch.save(model.state_dict(), model_name)

    # # delete models from previous epochs
    # print("Deleting models from previous epochs...")
    # for epoch in range(args.epochs):
    #     try:
    #         os.remove(f"{outdir}/model-{iteration}-epoch{epoch}.pt")
    #     except FileNotFoundError as ex:
    #         pass
    #         # print(f"No file exists {outdir}/model-{iteration}-epoch{epoch}.pt")
    #         # traceback.print_exc() 
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    all_outputs = np.array([])
    for ds in dataset.datasets:
        for key in ds.dfs_hashmap.keys():
            df = ds.dfs_hashmap[key]
            arr = df['angular_speed_z'].to_numpy()
            # print("len(arr)=", len(arr))
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            # print(f"Retrieved dataframe {key=}")
    dataset_moments = get_outputs_distribution(all_outputs)
    with open(f'./{outdir}/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.cumulative_sizes[-1]}\n"
                f"{args.epochs=}\n"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{lowest_loss=}\n"
                f"{device=}\n"
                f"{args.robustification=}\n"
                f"{args.noisevar=}\n"
                f"dataset_moments={dataset_moments}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")


if __name__ == '__main__':
    main()
