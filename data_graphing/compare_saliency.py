import sys
# sys.path.append("C:/Users/Meriel/Documents/GitHub/contextualvalidation")
sys.path.append("../models")
from DAVE2pytorch import *
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms
from pathlib import Path

# From our repository.
import saliency.core as saliency
import os
import pandas as pd
import random, string
import cv2
import time
import sys
sys.path.append("../models/")
sys.path.append( "../training")
from torch.autograd import Variable
from DatasetGenerator import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
from torchmetrics.image import VisualInformationFidelity
from skimage.metrics import structural_similarity as ssim
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='XRAISaliencyGeneration',
                    description='Generates XRAI heatmaps for each image in a trace using the model that steered the robot',
                    epilog='Generating XRAIs is computationally expensive so we generate them separately and then unpickle them for use in analyze_saliency.py')
    parser.add_argument('-w', '--modelweights', type=str, default="/p/rosbot/rosbotxl/models-yili/M1/model-DAVE2v3-2560x720-lr0.0001-100epoch-64batch-lossMSE-11Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur.pt")
    parser.add_argument('-r', '--resultsdir', type=str, default="./results-xrai/")
    parser.add_argument('-pt', '--passingtrace', type=str, default="/p/rosbot/rosbotxl/deployment-yili/M1/F4/")
    parser.add_argument('-px', '--passingxrais', type=str, default="/p/sdbb/ROSbot_data_collection/data_graphing/results-xrai/rosbot/rosbotxl/deployment-yili/M1/F4_XRAIs/")
    parser.add_argument('-pr', '--passingrange', type=str, default="rivian-00647,rivian-00876") #starts: rivian-00645 ends: rivian-00871
    parser.add_argument('-ft', '--failingtrace', type=str, default="/p/rosbot/rosbotxl/deployment-yili/M1/F3/")
    parser.add_argument('-fx', '--failingxrais', type=str, default="/p/sdbb/ROSbot_data_collection/data_graphing/results-xrai/rosbot/rosbotxl/deployment-yili/M1/F3_XRAIs/")
    parser.add_argument('-fr', '--failingrange', type=str, default="rivian-00694,rivian-00868") #starts: rivian-00694 ends: rivian-00868
    return parser.parse_args()

def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def timestr():
    localtime = time.localtime()
    return "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

def ShowImage(im, title='', ax=None, outfile="./output_xrai/", id=None):
    if ax is None:
        P.close("all")
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title, fontdict={"size": 18})
    P.savefig(f"{outfile}/showimg-{id}.jpg")

def ShowGrayscaleImage(im, title='', ax=None, outfile="./output_xrai/"):
    if ax is None:
        P.close("all")
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title, fontdict={"size": 18})
    P.colorbar()
    P.savefig(f"{outfile}/grayscale-{randstr()}.jpg")

def ShowHeatMap(im, title, ax=None, outfile="./output_xrai/", id=None):
    if ax is None:
        P.close("all")
        P.figure()
    P.axis('off')
    a = P.imshow(im, cmap='inferno')
    P.title(title, fontdict={"size": 12})
    P.colorbar()
    P.savefig(f"{outfile}/heatmap-{id}.jpg")
    return a

def ShowOverlay(image, overlay, title, ax=None, outfile="./output_xrai/", id=None):
    # np.array uint8
    if ax is None:
        P.close("all")
        P.figure()
    super_imposed_img = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    P.imshow(np.array(super_imposed_img), cmap='inferno')
    P.title(title, fontdict={"size": 18})
    P.savefig(f"{outfile}/heatmap-{id}-overlay.jpg")

def LoadImage(file_path, resize=(299, 299)):
    im = PIL.Image.open(file_path)
    im = im.resize(resize)
    im = np.asarray(im)
    return im

# transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def PreprocessImages(images, device=torch.device("cpu")):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    # images = transformer.forward(images)
    images = images.to(device)
    return images.requires_grad_(True)

def unpickle(picklefile):
    with open(picklefile, 'rb') as f:
        return pickle.load(f)

def parse_range(trace, rng):
    idx_start = [i for i in range(len(trace)) if trace[i] == rng.split(",")[0]+".pickle"][0]
    idx_end = [i for i in range(len(trace)) if trace[i] == rng.split(",")[1]+".pickle"][0]
    return idx_start, idx_end

from tqdm import tqdm
def main(args):
    outdir = f"./results-xrai-comparison2/"
    os.makedirs(outdir, exist_ok=True)
    passingtrace = os.listdir(args.passingxrais)
    passingtrace.sort()
    failingtrace = os.listdir(args.failingxrais)
    failingtrace.sort()
    passing_idx_start, passing_idx_end = parse_range(passingtrace, args.passingrange)
    failing_idx_start, failing_idx_end = parse_range(failingtrace, args.failingrange)
    passingtrace = passingtrace[passing_idx_start:passing_idx_end+1]
    failingtrace = failingtrace[failing_idx_start:failing_idx_end+1]
    # make passing and failing trace the same length for comparison's sake
    if len(passingtrace) > len(failingtrace):
        diff = len(passingtrace) - len(failingtrace)
        passingtrace = passingtrace[:-diff]
    elif len(passingtrace) < len(failingtrace):
        diff = len(failingtrace) - len(passingtrace)
        failingtrace = failingtrace[:-diff]
    print("Trace lengths: ", len(passingtrace), len(failingtrace))
    for i, pair in tqdm(enumerate(zip(passingtrace, failingtrace))):
        # print(f"PASS{pair[0].replace('.pickle', '')}-FAIL{pair[1].replace('.pickle', '')}")
        # load in each XRAI
        pass_hashmap = unpickle(args.passingxrais + pair[0])
        fail_hashmap = unpickle(args.failingxrais + pair[1])
        # dict_keys(['image', 'image_name', 'xrai', 'prediction', 'steering_input'])
        ROWS, COLS, UPSCALE_FACTOR = 2, 1, 10
        P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
        id = f"PASS{pair[0].replace('.pickle', '')}-FAIL{pair[1].replace('.pickle', '')}-plain"
        ShowImage(pass_hashmap['image'], title=f"COMPARISON \nPASS{pair[0].replace('.pickle', '')} (pred {pass_hashmap['prediction'].item():.3f})", ax=P.subplot(ROWS, COLS, 1), outfile=outdir, id=id)
        ShowImage(fail_hashmap['image'], title=f"FAIL{pair[1].replace('.pickle', '')} (pred {fail_hashmap['prediction'].item():.3f})", ax=P.subplot(ROWS, COLS, 2), outfile=outdir, id=id)
         
        # normalize heatmap scales
        normalization_max = max(np.max(pass_hashmap['xrai']), np.max(fail_hashmap['xrai']))
        normalization_min = min(np.min(pass_hashmap['xrai']), np.min(fail_hashmap['xrai']))
        pass_hashmap['xrai'][0][0] = normalization_max
        pass_hashmap['xrai'][0][1] = normalization_min
        fail_hashmap['xrai'][0][0] = normalization_max
        fail_hashmap['xrai'][0][1] = normalization_min

        # compare plain heatmaps
        ROWS, COLS, UPSCALE_FACTOR = 2, 1, 10
        P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
        id = f"PASS{pair[0].replace('.pickle', '')}-FAIL{pair[1].replace('.pickle', '')}"
        ShowHeatMap(pass_hashmap['xrai'], f"PASS{pair[0].replace('.pickle', '')}", ax=P.subplot(ROWS, COLS, 1), outfile=outdir, id=id)
        ShowHeatMap(fail_hashmap['xrai'], f"FAIL{pair[1].replace('.pickle', '')}", ax=P.subplot(ROWS, COLS, 2), outfile=outdir, id=id)

        # write overlaid normalized heatmaps
        ROWS, COLS, UPSCALE_FACTOR = 2, 1, 10
        P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
        colormap = plt.get_cmap('inferno')
        pass_heatmap = colormap(pass_hashmap['xrai'] / (normalization_max - normalization_min))
        pass_heatmap = np.delete(pass_heatmap, 3, 2) 
        fail_heatmap = colormap(fail_hashmap['xrai'] / (normalization_max - normalization_min))
        fail_heatmap = np.delete(fail_heatmap, 3, 2) 
        id = f"PASS{pair[0].replace('.pickle', '')}-FAIL{pair[1].replace('.pickle', '')}"
        ShowOverlay(pass_hashmap['image'], pass_heatmap.astype(np.float32), f"PASS{pair[0].replace('.pickle', '')}", ax=P.subplot(ROWS, COLS, 1), outfile=outdir, id=id)
        ShowOverlay(fail_hashmap['image'], fail_heatmap.astype(np.float32), f"FAIL{pair[1].replace('.pickle', '')}", ax=P.subplot(ROWS, COLS, 2), outfile=outdir, id=id)
        
        # compare diff'ed heatmaps
        pass_variance = np.var(pass_hashmap['xrai'])
        fail_variance = np.var(fail_hashmap['xrai'])
        ShowHeatMap(abs(pass_hashmap['xrai'] - fail_hashmap['xrai']), f"ABS. COMPARISON \nPASS{pair[0].replace('.pickle', '')} (pred {pass_hashmap['prediction'].item():.3f} var {pass_variance})\nFAIL{pair[1].replace('.pickle', '')} (pred {fail_hashmap['prediction'].item():.3f} var {fail_variance})", outfile=outdir, id=f"PASS{pair[0].replace('.pickle', '')}-FAIL{pair[1].replace('.pickle', '')}-abscomparison")
        ShowHeatMap(pass_hashmap['xrai'] - fail_hashmap['xrai'], f"SIGNED COMPARISON \nPASS{pair[0].replace('.pickle', '')} (pred {pass_hashmap['prediction'].item():.3f} var {pass_variance})\nFAIL{pair[1].replace('.pickle', '')} (pred {fail_hashmap['prediction'].item():.3f} var {fail_variance})", outfile=outdir, id=f"PASS{pair[0].replace('.pickle', '')}-FAIL{pair[1].replace('.pickle', '')}-comparison")

if __name__ == '__main__':
    args = parse_args()
    main(args)