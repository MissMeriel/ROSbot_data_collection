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
from DatasetGenerator import MultiDirectoryDataSequence
import time
import sys
sys.path.append("../models")
from Zach_DAVE2pytorch import DAVE2PytorchModel, DAVE2v1, DAVE2v2, DAVE2v3, Epoch

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--robustification", type=bool, default=True)
    parser.add_argument("--noisevar", type=int, default=15)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    return args


def characterize_steering_distribution(y_steering, generator):
    turning = []; straight = []
    for i in y_steering:
        if abs(i) < 0.1:
            straight.append(abs(i))
        else:
            turning.append(abs(i))
    # turning = [i for i in y_steering if i > 0.1]
    # straight = [i for i in y_steering if i <= 0.1]
    try:
        print("Moments of abs. val'd turning steering distribution:", generator.get_distribution_moments(turning))
        print("Moments of abs. val'd straight steering distribution:", generator.get_distribution_moments(straight))
    except Exception as e:
        print(e)
        print("len(turning)", len(turning))
        print("len(straight)", len(straight))


def main():
    start_time = time.time()
    input_shape = (2560, 720)  # Example input shape: width x height
    model = DAVE2v3(input_shape=input_shape)
    args = parse_arguments()
    print(args)
    dataset = MultiDirectoryDataSequence(args.dataset, image_size=(model.input_shape[::-1]), transform=Compose([ToTensor()]), \
                                         robustification=args.robustification, noise_level=args.noisevar) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples())
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    print("time to load dataset: {}".format(time.time() - start_time))

    iteration = f'{model._get_name()}-{input_shape[0]}x{input_shape[1]}-lr{args.lr}-{args.epochs}epoch-{args.batch}batch-lossMSE-{int(dataset.get_total_samples()/1000)}Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            batch_images = []
            batch_labels = []
            for samples in batch:
                for sample in samples:
                    batch_images.append(sample['image'])
                    batch_labels.append(sample['angular_speed_z'])

            # convert lists to tensors
            batch_images = torch.stack(batch_images).float().to(device)  # convert image list to tensor and move to device
            batch_labels = torch.stack(batch_labels).float().to(device)  # convert label list to tensor and move to device

            # wrap in variables for autograd
            batch_images = Variable(batch_images, requires_grad=True)
            batch_labels = Variable(batch_labels, requires_grad=False)

            optimizer.zero_grad()  # zero the parameter gradients

            # forward pass through the DNN
            outputs = model(batch_images)  # pass the batch of images to the DNN
            loss = F.mse_loss(outputs.flatten(), batch_labels)  # calculate the loss
            loss.backward()  # backpropagate the loss
            optimizer.step()  # update the model parameters
            running_loss += loss.item()
            if i % logfreq == logfreq - 1:
                print('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / logfreq))
                if (running_loss / logfreq) < lowest_loss:
                    print(f"New best model! MSE loss: {running_loss / logfreq}")
                    model_name = f"./model-{iteration}-best.pt"
                    print(f"Saving model to {model_name}")
                    torch.save(model, model_name)
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0
    print(f"Finished {epoch=}")
    model_name = f"/u/<your-computing-id>/ROSbot_data_collection/models/Dave2-Keras/model-{iteration}-epoch{epoch}.pt"
    print(f"Saving model to {model_name}")
    torch.save(model, model_name)
        # if loss < 0.002:
        #     print(f"Loss at {loss}; quitting training...")
        #     break
    print('Finished Training')

    # save model
    # torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    model_name = f'/u/<your-computing-id>/ROSbot_data_collection/models/Dave2-Keras/model-{iteration}.pt'
    torch.save(model.state_dict(), model_name)

    # delete models from previous epochs
    print("Deleting models from previous epochs...")
    for epoch in range(args.epochs):
        os.remove(f"/u/<your-computing-id>/ROSbot_data_collection/models/Dave2-Keras/model-{iteration}-epoch{epoch}.pt")
    print(f"Saving model to {model_name}")
    print("All done :)")
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train))
    # save metainformation about training
    with open(f'/u/<your-computing-id>/ROSbot_data_collection/models/Dave2-Keras/model-{iteration}-metainfo.txt', "w") as f:
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