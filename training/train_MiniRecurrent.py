import numpy as np
import argparse
import pandas as pd
import matplotlib.image as mpimg
from torch.autograd import Variable

# import h5py
import os
# from PIL import Image
# import PIL
import matplotlib.pyplot as plt
# import csv
from DatasetGenerator_Recurrent import MultiDirectoryDataSequence
import time
import sys
sys.path.append("../models")
from DAVE2pytorch import *
from MiniRecurrent import *

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import data
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
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--recurrence_length", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--slurmid", type=int, default=0)
    parser.add_argument("--archid", type=str, default="MiniRNN")
    parser.add_argument("--lossfxn", type=str, default="MSE")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    print(args)
    start_time = time.time()
    input_shape = (2560, 720)  # Example input shape: width x height
    if args.archid == "DAVE2v1":
        model = DAVE2v1(input_shape=input_shape)
    elif args.archid == "DAVE2v2":
        model = DAVE2v2(input_shape=input_shape)
    elif args.archid == "DAVE2v3":
        model = DAVE2v3(input_shape=input_shape)
    elif args.archid == "DAVE2v3Norm":
        model = DAVE2v3Norm(input_shape=input_shape)
    elif args.archid == "Epoch":
        model = Epoch(input_shape=input_shape)
    elif args.archid == "MiniRNN":
        # rnn_type, ntoken, ninp, nhid, nlayers
        model = MiniRNN(input_shape=input_shape, rnn_hidden_size=args.recurrence_length, nlayers=args.batch, dropout=0.5, tie_weights=False)

    if args.normalize:
        transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = Compose([ToTensor()])
    dataset = MultiDirectoryDataSequence(args.dataset, image_size=(model.input_shape[::-1]), transform=transform,\
                                         robustification=args.robustification, noise_level=args.noisevar, recurrence_length=args.recurrence_length)
    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples(), flush=True)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, worker_init_fn=worker_init_fn)
    print(f"Time to load dataset: {(time.time() - start_time):.1f}s", flush=True)
    outdir = f"./training-output/{model._get_name()}-NOSHUFFLE-{randstr()}-{timestr()}-{args.slurmid}/"
    os.makedirs(outdir, exist_ok=True)
    iteration = f'{model._get_name()}-NOSHUFFLE-{input_shape[0]}x{input_shape[1]}-loss{args.lossfxn}-aug{args.robustification}-converge{args.convergence}-norm{args.normalize}-{args.epochs}epoch-{args.batch}batch-{int(dataset.get_total_samples()/1000)}Ksamples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model.train()
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20
    if args.lossfxn == "L1":
        lossfxn = nn.L1Loss()
    else:
        lossfxn = nn.MSELoss()
    for epoch in range(args.epochs):
        epoch_start_time = time.time() # Record start time for the epoch
        running_loss = 0.0
        sampled_loss = np.ones(10)
        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image name'].float().to(device)
            y = hashmap['angular_speed_z'].float().to(device)
            current_rvec = hashmap['recurrent_vec'][0].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)
            current_rvec = Variable(current_rvec, requires_grad=True)
            # print(f"Batch {i} {current_rvec.shape=}")
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, rvec = model(x, current_rvec)
            loss = lossfxn(outputs, y) #+ lossfxn(current_rvec, rvec)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq))
                sampled_loss = np.roll(sampled_loss, 1)
                sampled_loss[0] = running_loss / logfreq
                if epoch > 5 and (running_loss / logfreq) < lowest_loss:
                    print(f"New best model! {args.lossfxn} Loss: {running_loss / logfreq}", flush=True)
                    model_name = f"{outdir}/model-{iteration}-best.pt"
                    print(f"Saving model to {model_name}", flush=True)
                    torch.save(model, model_name)
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0
        epoch_end_time = time.time()  # Record end time for the epoch
        epoch_duration = epoch_end_time - epoch_start_time # Record total time for the epoch
        print(f"Finished Epoch {epoch + 1}\nEpoch Duration: {epoch_duration:.1f}s", flush=True)
        sys.stdout.flush()
        model_name = f"{outdir}/model-{iteration}-epoch{epoch}.pt"
        print(f"Saving model to {model_name}")
        torch.save(model, model_name)
        print(f"{np.var(sampled_loss)=}", flush=True)
        if args.convergence and np.var(sampled_loss) < 0.0001:
            print(f"Loss converged at {loss} (variance {np.var(sampled_loss):.4f}); quitting training...", flush=True)
            break
    print('Finished Training')

    # save model
    # torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    model_name = f'{outdir}/model-{iteration}.pt'
    torch.save(model.state_dict(), model_name)

    # delete models from previous epochs
    print("Deleting models from previous epochs...")
    for epoch in range(args.epochs):
        os.remove(f"{outdir}/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    with open(f'./{outdir}/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.get_total_samples()}\n"
                f"{args.epochs=}\n"
                f"{args.lr=}\n"
                f"{args.batch=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{args.robustification=}\n"
                f"{args.noisevar=}\n"
                f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")


if __name__ == '__main__':
    main()
