import sys
sys.path.append("C:/Users/Meriel/Documents/GitHub/contextualvalidation")
sys.path.append("C:/Users/Meriel/Documents/GitHub/contextualvalidation/DAVE2")
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

from torch.autograd import Variable
from UUSTDatasetGenerator import MultiDirectoryDataSequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize
from torchmetrics.image import VisualInformationFidelity
from skimage.metrics import structural_similarity as ssim
import pickle

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
    P.savefig(f"{outfile}/showimg-{id}-{randstr()}.jpg")

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
    P.title(title, fontdict={"size": 18})
    P.colorbar()
    P.savefig(f"{outfile}/heatmap-{id}-{randstr()}.jpg")
    return a

def ShowOverlay(image, overlay, title, ax=None, outfile="./output_xrai/", id=None):
    # np.array uint8
    if ax is None:
        P.close("all")
        P.figure()
    super_imposed_img = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    P.imshow(np.array(super_imposed_img), cmap='inferno')
    P.title(title, fontdict={"size": 18})
    P.savefig(f"{outfile}/overlay-heatmap-{randstr()}-{id}.jpg")

def LoadImage(file_path, resize=(299, 299)):
    im = PIL.Image.open(file_path)
    im = im.resize(resize)
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device("cpu")
weightpath = "weights/model-DAVE2extrasmoothgrad-1000epoch-145Ksamples-epoch018-best046.pt"
model = torch.load(weightpath, map_location=device).eval()
model = model.requires_grad_(True)
basic_transform = Compose([ToTensor()])
# Register hooks for Grad-CAM, which uses the last convolution layer
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


# VAL_DATASET = "F:/supervised-transformation-dataset-alltransforms31FULL-V/"
# VAL_DATASET = "F:/supervised-transformation-dataset-indistribution/"
VAL_DATASET = "F:/supervised-transformation-dataset-alltransforms3/"
dataset1 = MultiDirectoryDataSequence(VAL_DATASET, image_size=(model.input_shape[::-1]),
                                     transform=Compose([ToTensor()]), \
                                     robustification=False, noise_level=0,
                                     sample_id="STEERING_INPUT")

# dataset1.reverse()

print("Moments of distribution:", dataset1.get_outputs_distribution(), flush=True)
print("Total samples:", dataset1.get_total_samples(), flush=True)

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
xrai_outdir = Path(VAL_DATASET)
xrai_outdir = VAL_DATASET[:-1] + "_XRAIs2/"
os.makedirs(xrai_outdir, exist_ok=True)
start_time = time.time()
for i1, hashmap1 in enumerate(trainloader1, 0):
    # print(f"{hashmap1=}")
    im_tensor1 = hashmap1['image_base'].to(device, dtype=torch.float)
    y1 = hashmap1['steering_input'].float().to(device)
    img_name1 = hashmap1['img_name'].numpy()[0].tolist()
    img_name1 = ''.join(chr(int(i)) for i in img_name1[0]).rstrip('\x00')
    # print(img_name1)
    # im_tensor1_processed = model.process_image(im_tensor1).to(device, dtype=torch.float)
    predictions1 = model(im_tensor1)
    predictions1 = predictions1.detach().cpu().numpy()
    im1 = im_tensor1.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)[0]
    xrai_attributions1 = xrai_object.GetMask(im1, call_model_function, call_model_args, batch_size=20)
    write_xrai({"image": im1, "image_name": Path(img_name1).parts[-2:], "xrai": xrai_attributions1, "prediction": predictions1, "steering_input": y1}, Path(img_name1), xrai_outdir)
    if i1 % 500 == 0 and i1 > 0:
        elapsed_time = time.time() - start_time
        print(f"Finished XRAIs for {i1} samples (avg. {elapsed_time/i1:.1f}sec/sample)")

