# https://pytorch.org/hub/pytorch_vision_shufflenet_v2/

import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np

class ShuffleNetSteer(nn.Module):
    def __init__(self, input_shape=(150, 200)):
        super().__init__()
        self.input_shape = input_shape
        self.preprocess = transforms.Compose([
            transforms.Resize(input_shape),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
        self.model.eval()
        print(self.model)
        self.featuresize = self.model.conv5(self.model.stage4(self.model.stage3(self.model.stage2(self.model.maxpool(self.model.conv1(torch.zeros(1, 3, *self.input_shape)))))))
            # ()))))))
        self.featuresize = np.prod(self.featuresize.shape)
        print(self.featuresize)
        self.head = nn.Sequential(nn.Linear(in_features=self.featuresize, out_features=1, bias=True))
        self.dropout = nn.Dropout()

    def forward(self, x):
        # original ShuffleNet layers
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        # meriel's ShuffleNetSteer additions
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) #x.flatten()
        print(f"{x.shape=} {self.featuresize=}")
        # x = self.model.fc(x)
        x = self.head(x)
        x = torch.tanh(x)
        return x

def main():
    input_shape=(100,100) #(512,144)
    model = ShuffleNetSteer(input_shape=input_shape)
    print(model)

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    # Here’s a sample execution.

    # Download an example image from the pytorch website
    input_image = Image.open("/p/sdbb/ROSbot_data_collection/failure-catalog/M1/F1/rivian-00534.jpg")
    input_tensor = model.preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(f"{output.shape=}")
    print(f"{output.item()}")



if __name__ == '__main__':
    main()