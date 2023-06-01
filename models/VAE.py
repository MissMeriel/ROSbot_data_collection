import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToPILImage, ToTensor
from scipy.stats import truncnorm
import cv2

# based off of:
#   https://github.com/swa112003/DistributionAwareDNNTesting
#   https://github.com/AntixK/PyTorch-VAE

class VAE(nn.Module):
    def __init__(self, input_shape=(150, 200), in_channels=3, latent_dim=20):
        super().__init__()
        # # instantiate VAE model
        # outputs = decoder(encoder(input_tensor)[2])
        # vae = Model(input_tensor, outputs, name='vae_svhn')
        # # vae.summary()
        self.input_shape = input_shape
        self.encoder = self.Encoder(input_shape)
        self.decoder = self.Decoder(input_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def load(self, path="test-model.pt"):
        return torch.load(path)

    # process PIL image to Tensor
    # @classmethod
    def process_image(self, image, transform=Compose([ToTensor()])):
        # image = image.resize((self.input_shape[1], self.input_shape[0]), Image.ANTIALIAS)
        # image = cv2.resize(image, self.input_shape)
        # add a single dimension to the front of the matrix -- [...,None] inserts dimension in index 1
        # image = np.array(image)[None]#.reshape(1, self.input_shape[0], self.input_shape[1], 3)
        # use transpose instead of reshape -- reshape doesn't change representation in memory
        # image = image.transpose((0,3,1,2))
        # ToTensor() normalizes data between 0-1 but torch.from_numppy just casts to Tensor
        # if transform:
        # image = torch.from_numpy(image)/ 255.0 #127.5-1.0 #transform(image)
        # print(f"{image.shape=}")
        image = transform(np.asarray(image))[None]
        return image  # .permute(0,3,1,2)#(2, 1, 0)
    
    class Encoder(nn.Module):
        def __init__(self, input_shape=(150, 200)):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 5, stride=2)
            self.conv2 = nn.Conv2d(8, 16, 5, stride=1)
            self.conv3 = nn.Conv2d(16, 32, 5, stride=2)
            self.conv4 = nn.Conv2d(32, 64, 3, stride=1)
            self.conv5 = nn.Conv2d(64, 64, 3, stride=2)
            # # build encoder model
            # x = Conv2D(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(input_tensor)
            # x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
            # x = Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
            # x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
            # x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
            # # shape info needed to build decoder model
            # shape = K.int_shape(x)
            # # generate latent vector Q(z|X)
            # x = Flatten()(x)
            # z_mean = Dense(latent_dim, name='z_mean')(x)
            # z_log_var = Dense(latent_dim, name='z_log_var')(x)
            # # use reparameterization trick to push the sampling out as input
            # # note that "output_shape" isn't necessary with the TensorFlow backend
            # z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
            # # instantiate encoder model
            # # encoder = Model(input_tensor, [z_mean, z_log_var, z], name='encoder')

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.conv4(x)
            x = F.relu(x)
            x = self.conv5(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, input_shape=(150, 200)):
            super().__init__()
            # build decoder model
            self.input_shape = (1, input_shape[0], input_shape[1], 3)
            size = self.input_shape[1]*self.input_shape[2]*self.input_shape[3]
            self.lin1 = nn.Linear(in_features=3, out_features=size, bias=True)
            self.conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2))
            self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5))
            self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(2,2))
            self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5))
            self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(2,2))
            self.pos_mean = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5))
            self.pos_log_var = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5,5))
            
            # latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
            # x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
            # x = Reshape((shape[1], shape[2], shape[3]))(x)
            # 
            # x = Conv2DTranspose(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
            # x = Conv2DTranspose(64, (5, 5), padding='same', activation='relu')(x)
            # x = Conv2DTranspose(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
            # x = Conv2DTranspose(16, (5, 5), padding='same', activation='relu')(x)
            # x = Conv2DTranspose(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
            # pos_mean = Conv2DTranspose(3, (5, 5), padding='same', name='pos_mean')(x)
            # pos_log_var = Conv2DTranspose(3, (5, 5), padding='same', name='pos_log_var')(x)
            # 
            # # instantiate decoder model
            # decoder = Model(latent_inputs, [pos_mean, pos_log_var], name='decoder')
            # # decoder.summary()
            
        def forward(self, x):
            x = self.lin1(x)
            x = torch.reshape(x, self.input_shape)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.conv4(x)
            x = F.relu(x)
            x = self.conv5(x)
            x = F.relu(x)
            x1 = self.pos_mean(x)
            x2 = self.pos_log_var(x)
            return [x1, x2]