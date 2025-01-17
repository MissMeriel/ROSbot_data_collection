# https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import numpy as np

class MobileNetSteer(nn.Module):
    def __init__(self, input_shape=(150, 200)):
        super().__init__()
        self.input_shape = input_shape
        self.preprocess = transforms.Compose([
            transforms.Resize(input_shape),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.eval()
        print(self.model)
        self.size = np.prod(self.model.features(torch.zeros(1, 3, *self.input_shape)).shape)
        self.head = nn.Sequential(nn.Linear(in_features=self.size, out_features=1, bias=True))
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.model.features(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1) #x.flatten()
        # print(f"{x.shape=} {self.size=}")
        x = self.head(x)
        x = torch.tanh(x)
        return x



if __name__ == '__main__':
    mns_model = MobileNetSteer()
    input_image = Image.open("/p/sdbb/ROSbot_data_collection/failure-catalog/M1/F1/rivian-00534.jpg")
    input_tensor = mns_model.preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        mns_model.to('cuda')

    with torch.no_grad():
        output = mns_model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    print(f"{output.shape=}")
    print(output)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)