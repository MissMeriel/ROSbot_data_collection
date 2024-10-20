import sys
sys.path.append("../models")
import numpy as np
import os
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
from DAVE2pytorch import *
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
from utils import *


from DroNet import DronetTorch
# from dronet_datasets import DronetDataset # https://github.com/peasant98/Dronet-Pytorch/blob/master/dronet_datasets.py

import torch
from DatasetGenerator_DroNet import MultiDirectoryDataSequence



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
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--slurmid", type=int, default=0)
    parser.add_argument("--lossfxn", type=str, default="MSE")
    args = parser.parse_args()
    return args


def getModel(img_dims, img_channels, output_dim, weights_path):
    '''
      Initialize model.

      ## Arguments

        `img_dims`: Target image dimensions.

        `img_channels`: Target image channels.

        `output_dim`: Dimension of model output.
        
        `weights_path`: Path to pre-trained model.

      ## Returns
        `model`: the pytorch model
    '''
    model = DronetTorch(img_dims, img_channels, output_dim)
    # if weights path exists...
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path))
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model

def trainModel(model: DronetTorch, 
                epochs, batch_size, steps_save, k, args):
    '''
    trains the model.

    ## parameters:
    

    '''
    model.to(model.device)

    model.train()
    # create dataloaders for validation and training
    # training_dataset = DronetDataset('data/collision/collision_dataset', 'training', augmentation=False
    #                                     )
    # validation_dataset = DronetDataset('data/collision/collision_dataset', 'validation',
    #                                     augmentation=False)

    # training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, 
    #                                         shuffle=True, num_workers=4)
    # validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, 
    #                                         shuffle=False, num_workers=4)
    input_shape = (2560, 720) 
    if args.normalize:
        transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = Compose([ToTensor()])
    dataset = MultiDirectoryDataSequence(args.dataset, image_size=(input_shape[::-1]), transform=transform,\
                                         robustification=args.robustification, noise_level=args.noisevar)
    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())
    print("Total samples:", dataset.get_total_samples(), flush=True)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    training_dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, worker_init_fn=worker_init_fn)
    # adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch_loss = np.zeros((epochs, 2))
    for epoch in range(epochs):
        # scale the weights on the loss and the epoch number
        train_losses = []
        validation_losses = []
        # rip through the dataset
        other_val = (1 - torch.exp(torch.Tensor([-1*model.decay * (epoch-10)]))).float().to(model.device)
        model.beta = torch.max(torch.Tensor([0]).float().to(model.device), other_val)
        for batch_idx, hm in enumerate(training_dataloader):
            img = hm["image name"]
            steer_true = hm["angular_speed_z"]
            coll_true = hm["collision_probability"]
            img_cuda = img.float().to(model.device)
            steer_pred, coll_pred = model(img_cuda)
            # get loss, perform hard mining
            steer_true = steer_true.to(model.device)
            coll_true  = coll_true.to(model.device)
            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
            # backpropagate loss
            loss.backward()
            # optimizer step
            optimizer.step()
            # zero gradients to prevestepnt accumulation, for now
            optimizer.zero_grad()
            # print(f'loss: {loss.item()}')
            train_losses.append(loss.item())
            print(f'Training Images Epoch {epoch}: {batch_idx * batch_size}')
        train_loss = np.array(train_losses).mean()
        if epoch % steps_save == 0:
            print('Saving results...')

            weights_path = os.path.join('models', f'weights_{epoch:03d}.pth')
            torch.save(model.state_dict(), weights_path)
        # evaluate on validation set
        # for batch_idx, (img, steer_true, coll_true) in enumerate(validation_dataloader):
        #     img_cuda = img.float().to(model.device)
        #     steer_pred, coll_pred = model(img_cuda)
        #     steer_true = steer_true.to(model.device)
        #     coll_true = coll_true.to(model.device)
        #     loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
        #     validation_losses.append(loss.item())
        #     print(f'Validation Images: {batch_idx * 4}')

        # validation_loss = np.array(validation_losses).mean()
        epoch_loss[epoch, 0] = train_loss 
        epoch_loss[epoch, 1] = validation_loss
        # Save training and validation losses.
    # save final results
    weights_path = os.path.join('models', 'dronet_trained.pth')
    torch.save(model.state_dict(), weights_path)
    np.savetxt(os.path.join('models', 'losses.txt'), epoch_loss)

if __name__ == "__main__":
    dronet = getModel((224,224), 3, 1, None)
    print(dronet)
    args = parse_arguments()
    trainModel(dronet, 150, 16, 5, 8, args)