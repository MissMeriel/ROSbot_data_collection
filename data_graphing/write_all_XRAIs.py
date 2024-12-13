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


# For saliency analysis:
# The glassy conference room area (the part where the grey room divider thing used to be during the summer)
# Successful trace (from M1F4):
# path:  /p/rosbot/rosbotxl/deployment-yili/M1/F4
# trace starts: rivian-00645 ends: rivian-00871
# Failing trace (from M1F3):
# path:  /p/rosbot/rosbotxl/deployment-yili/M1/F3
# trace starts: rivian-00694 ends: rivian-00868
# M1 path: /p/rosbot/rosbotxl/models-yili/M1

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='XRAISaliencyGeneration',
                    description='Generates XRAI heatmaps for each image in a trace using the model that steered the robot',
                    epilog='Generating XRAIs is computationally expensive so we generate them separately and then unpickle them for use in analyze_saliency.py')
    parser.add_argument('-w', '--modelweights', type=str, default="/p/rosbot/rosbotxl/models-yili/M1/model-DAVE2v3-2560x720-lr0.0001-100epoch-64batch-lossMSE-11Ksamples-INDUSTRIALandHIROCHIandUTAH-noiseflipblur.pt")
    parser.add_argument('-r', '--resultsdir', type=str, default="./results-xrai/")
    parser.add_argument('-t', '--trace', type=str, default="/p/rosbot/rosbotxl/deployment-yili/M1/F4/")
    return parser.parse_args()

def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def timestr():
    localtime = time.localtime()
    return "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

# def ShowImage(im, title='', ax=None, outfile="./output_xrai/", id=None):
#     if ax is None:
#         P.close("all")
#         P.figure()
#     P.axis('off')
#     P.imshow(im)
#     P.title(title, fontdict={"size": 18})
#     P.savefig(f"{outfile}/showimg-{id}-{randstr()}.jpg")

# def ShowGrayscaleImage(im, title='', ax=None, outfile="./output_xrai/"):
#     if ax is None:
#         P.close("all")
#         P.figure()
#     P.axis('off')
#     P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
#     P.title(title, fontdict={"size": 18})
#     P.colorbar()
#     P.savefig(f"{outfile}/grayscale-{randstr()}.jpg")

# def ShowHeatMap(im, title, ax=None, outfile="./output_xrai/", id=None):
#     if ax is None:
#         P.close("all")
#         P.figure()
#     P.axis('off')
#     a = P.imshow(im, cmap='inferno')
#     P.title(title, fontdict={"size": 18})
#     P.colorbar()
#     P.savefig(f"{outfile}/heatmap-{id}-{randstr()}.jpg")
#     return a

# def ShowOverlay(image, overlay, title, ax=None, outfile="./output_xrai/", id=None):
#     # np.array uint8
#     if ax is None:
#         P.close("all")
#         P.figure()
#     super_imposed_img = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
#     P.imshow(np.array(super_imposed_img), cmap='inferno')
#     P.title(title, fontdict={"size": 18})
#     P.savefig(f"{outfile}/overlay-heatmap-{randstr()}-{id}.jpg")

# def LoadImage(file_path, resize=(299, 299)):
#     im = PIL.Image.open(file_path)
#     im = im.resize(resize)
#     im = np.asarray(im)
#     return im

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

def main(args):
    parentdir = f"./{args.resultsdir}/passfail-{timestr()}-{randstr()}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device("cpu")
    input_shape = (2560, 720)
    # model = torch.load(args.modelweights, map_location=device).eval()
    # try:
    model = DAVE2v3(input_shape=input_shape)
    model.load_state_dict(torch.load(args.modelweights, map_location=torch.device('cpu')))
    # except TypeError as e:
    #     try:
    #         model = torch.load(args.modelweights, map_location=torch.device('cpu'))
    #     except TypeError as e:
    #         print(e)
    model = model.requires_grad_(True)
    basic_transform = Compose([ToTensor()])
    # Register hooks for Grad-CAM, which uses the last convolution layer
    model.init_features()
    conv_layer = model.features
    conv_layer_outputs = {}
    def conv_layer_forward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().cpu().numpy()
    def conv_layer_backward(m, i, o):
        # move the RGB dimension to the last dimension
        conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().cpu().numpy()
    conv_layer.register_forward_hook(conv_layer_forward)
    conv_layer.register_full_backward_hook(conv_layer_backward)

    class_idx_str = 'class_idx_str'
    def call_model_function(images, call_model_args=None, expected_keys=[saliency.base.INPUT_OUTPUT_GRADIENTS]):
        images = PreprocessImages(images)
        # print(f"{images.shape=}")
        # target_class_idx = call_model_args[class_idx_str]
        output = model(images.to(device))
        # m = torch.nn.Softmax(dim=1)
        # output = m(output)
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            # outputs = output[:,target_class_idx]
            # grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            grads = torch.autograd.grad(output, images, grad_outputs=torch.ones_like(output))
            grads = torch.movedim(grads[0], 1, 3)
            gradients = grads.detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            one_hot = output #torch.zeros_like(output)
            # one_hot[:,target_class_idx] = 1
            model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return conv_layer_outputs

    def write_xrai(xrai, imgname, xrai_outdir):
        imgname = "/".join(imgname.parts[-2:])
        os.makedirs(xrai_outdir + "/".join(Path(imgname).parts[:-1]), exist_ok=True)
        writefile = xrai_outdir + imgname.replace(".jpg", ".pickle")
        writefile = writefile.replace("\\", "/")
        writefile = writefile.rstrip('\x00')
        with open(writefile, "wb") as f:
            pickle.dump(xrai, f)


    dataset1 = DataSequence( args.trace, transform=Compose([ToTensor()]))
    dataset2 = DataSequence( args.failingtrace, transform=Compose([ToTensor()]))

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    BATCH_SIZE = 1
    trainloader1 = DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn, num_workers=0)

    agg_img = np.zeros((108, 192, 3))
    agg_grad_smoothgrad = np.zeros((108, 192))
    agg_grad_xrai = np.zeros((108, 192))
    vif = VisualInformationFidelity()
    vifs, ssims, preds = [], [], []
    call_model_args = {}
    xrai_object = saliency.XRAI()
    integrated_gradients = saliency.IntegratedGradients()
    # xrai_outdir = Path(args.trace)
    # xrai_outdir = args.trace[:-1] + "_XRAIs/"
    xrai_outdir = args.resultsdir + args.trace[3:-1] + "_XRAIs/"
    print(f"Results writing to {xrai_outdir}")
    os.makedirs(xrai_outdir , exist_ok=True)
    start_time = time.time()
    for i1, hashmap1 in enumerate(trainloader1, 0):
        # print(f"{hashmap1=}")
        im_tensor1 = hashmap1['image'].to(device, dtype=torch.float)
        y1 = hashmap1['angular_speed_z'].float().to(device)
        img_name1 = hashmap1['image_name'][0]
        # im_tensor1_processed = model.process_image(im_tensor1).to(device, dtype=torch.float)
        predictions1 = model(im_tensor1)
        predictions1 = predictions1.detach().cpu().numpy()
        im1 = im_tensor1.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)[0]
        xrai_attributions1 = xrai_object.GetMask(im1, call_model_function, call_model_args, batch_size=20)
        write_xrai({"image": im1, "image_name": Path(img_name1).parts[-2:], "xrai": xrai_attributions1, "prediction": predictions1, "steering_input": y1}, Path(img_name1), xrai_outdir)
        if i1 % 500 == 0 and i1 > 0:
            elapsed_time = time.time() - start_time
            print(f"Finished XRAIs for {i1} samples (avg. {elapsed_time/i1:.1f}sec/sample)")

if __name__ == '__main__':
    args = parse_args()
    main(args)