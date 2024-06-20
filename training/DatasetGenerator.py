import numpy as np
import os, cv2, csv
# from DAVE2 import DAVE2Model
# from DAVE2pytorch import DAVE2PytorchModel
import kornia

from PIL import Image
import copy
from scipy import stats
import torch.utils.data as data
from pathlib import Path
import skimage.io as sio
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import random

from torchvision.transforms import Compose, ToTensor, PILToTensor, functional as transforms
# from io import BytesIO
# import skimage

def stripleftchars(s):
    # print(f"{s=}")
    for i in range(len(s)):
        if s[i].isnumeric():
            return s[i:]
    return -1

class DataSequence(data.Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform

        image_paths = []
        for p in Path(root).iterdir():
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                image_paths.append(p)
        image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
        self.image_paths = image_paths
        # print(f"{self.image_paths=}")
        self.df = pd.read_csv(f"{self.root}/data_cleaned.csv")
        self.cache = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img_name = self.image_paths[idx]
        image = sio.imread(img_name)

        df_index = self.df.index[self.df['image name'] == img_name.name]
        y_thro = self.df.loc[df_index, 'linear_speed_x'].array[0]
        y_steer = self.df.loc[df_index, 'angular_speed_z'].array[0]
        y = [y_steer, y_thro]
        # torch.stack(y, dim=1)
        y = torch.tensor(y_steer)

        # plt.title(f"steering_input={y_steer.array[0]}")
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.01)

        if self.transform:
            image = self.transform(image).float()
        # print(f"{img_name.name=} {y_steer=}")
        # print(f"{image=}")
        # print(f"{type(image)=}")
        # print(self.df)
        # print(y_steer.array[0])

        # sample = {"image": image, "steering_input": y_steer.array[0]}
        sample = {"image name": image, "angular_speed_z": y}

        self.cache[idx] = sample
        return sample

class MultiDirectoryDataSequence(data.Dataset):
    def __init__(self, root, image_size=(100,100), transform=None, robustification=False, noise_level=10):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.size = 0
        self.image_size = image_size
        image_paths_hashmap = {}
        all_image_paths = []
        self.dfs_hashmap = {}
        self.dirs = []
        # marker = "_YES"
        marker = "collection"
        for p in Path(root).iterdir():
            if p.is_dir() and marker in str(p): #"_NO" not in str(p) and "YQWHF3" not in str(p):
                self.dirs.append("{}/{}".format(p.parent,p.stem.replace(marker, "")))
                image_paths = []
                # print(f"testing testing testing!")
                try:
                    self.dfs_hashmap[f"{p}"] = pd.read_csv(f"{p}/data_cleaned.csv")
                    # add to debug
                    print(f"Found data_cleaned.csv in {p.parent}/{p.stem.replace(marker, '')}")
                except FileNotFoundError as e:
                    print(f"{e} \nNo data_cleaned.csv in directory {p.parent}/{p.stem.replace(marker, '')}")
                    continue
                for pp in Path(p).iterdir():
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"] and "collection_trajectory" not in pp.name:
                        image_paths.append(pp)
                        all_image_paths.append(pp)
                if image_paths:
                    image_paths.sort(key=lambda p: int(stripleftchars(p.stem)))
                    image_paths_hashmap[p] = copy.deepcopy(image_paths)
                    self.size += len(image_paths)
                else:
                    print(f"No valid images found in directory: {p.parent}/{p.stem.replace(marker, '')}")
        print("Finished intaking image paths!")
        self.image_paths_hashmap = image_paths_hashmap
        self.all_image_paths = all_image_paths
        # self.df = pd.read_csv(f"{self.root}/data_cleaned.csv")
        self.cache = {}
        self.robustification = robustification
        self.noise_level = noise_level

    def get_total_samples(self):
        return self.size

    def get_directories(self):
        return self.dirs
        
    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            if self.robustification:
                sample = self.cache[idx]
                y_steer = sample["angular_speed_z"]
                image = copy.deepcopy(sample["image name"])
                if random.random() > 0.5:
                    # flip image
                    image = torch.flip(image, (2,))
                    y_steer = -sample["angular_speed_z"]
                if random.random() > 0.5:
                    # blur
                    gauss = kornia.filters.GaussianBlur2d((3,3), (1.5, 1.5))
                    image = gauss(image[None])[0]
                image = torch.clamp(image + (torch.randn(*image.shape) / self.noise_level), 0, 1)
                return {"image name": image, "angular_speed_z": y_steer, "linear_speed_x": sample["linear_speed_x"], "all": torch.FloatTensor([y_steer, sample["linear_speed_x"]])}
            else:
                return self.cache[idx]
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)
        image = image.resize(self.image_size)
        # image = cv2.imread(img_name.__str__())
        # image = cv2.resize(image, self.image_size) / 255
        # image = self.fisheye(image)
        orig_image = self.transform(image)
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df['image name'] == img_name.name]
        orig_y_steer = df.loc[df_index, 'angular_speed_z'].item()
        y_throttle = df.loc[df_index, 'linear_speed_x'].item()
        y_steer = copy.deepcopy(orig_y_steer)
        if self.robustification:
            image = copy.deepcopy(orig_image)
            if random.random() > 0.5:
                # flip image
                image = torch.flip(image, (2,))
                y_steer = -orig_y_steer
            if random.random() > 0.5:
                # blur
                gauss = kornia.filters.GaussianBlur2d((5, 5), (5.5, 5.5))
                image = gauss(image[None])[0]
                # image = kornia.filters.blur_pool2d(image[None], 3)[0]
                # image = kornia.filters.max_blur_pool2d(image[None], 3, ceil_mode=True)[0]
                # image = kornia.filters.median_blur(image, (3, 3))
                # image = kornia.filters.median_blur(image, (10, 10))
                # image = kornia.filters.box_blur(image, (3, 3))
                # image = kornia.filters.box_blur(image, (5, 5))
                # image = kornia.resize(image, image.shape[2:])
                # plt.imshow(image.permute(1,2,0))
                # plt.pause(0.01)
            image = torch.clamp(image + (torch.randn(*image.shape) / self.noise_level), 0, 1)

        else:
            t = Compose([ToTensor()])
            image = t(image).float()
            # image = torch.from_numpy(image).permute(2,0,1) / 127.5 - 1

        # vvvvvv uncomment below for value-image debugging vvvvvv
        # plt.title(f"{img_name}\nsteering_input={y_steer.array[0]}", fontsize=7)
        # plt.imshow(image)
        # plt.show()
        # plt.pause(0.01)

        sample = {"image name": image, "angular_speed_z": torch.FloatTensor([y_steer]), "linear_speed_x": torch.FloatTensor([y_throttle]), "all": torch.FloatTensor([y_steer, y_throttle])}
        orig_sample = {"image name": orig_image, "angular_speed_z": torch.FloatTensor([orig_y_steer]), "linear_speed_x": torch.FloatTensor([y_throttle]), "all": torch.FloatTensor([orig_y_steer, y_throttle])}
        self.cache[idx] = orig_sample
        return sample

    def get_outputs_distribution(self):
        all_outputs = np.array([])
        for key in self.dfs_hashmap.keys():
            df = self.dfs_hashmap[key]
            arr = df['angular_speed_z'].to_numpy()
            # print("len(arr)=", len(arr))
            all_outputs = np.concatenate((all_outputs, arr), axis=0)
            # print(f"Retrieved dataframe {key=}")
        all_outputs = np.array(all_outputs)
        moments = self.get_distribution_moments(all_outputs)
        return moments

    ##################################################
    # ANALYSIS METHODS
    ##################################################

    # Moments are 1=mean 2=variance 3=skewness, 4=kurtosis
    def get_distribution_moments(self, arr):
        moments = {}
        moments['shape'] = np.asarray(arr).shape
        moments['mean'] = np.mean(arr)
        moments['median'] = np.median(arr)
        moments['var'] = np.var(arr)
        moments['skew'] = stats.skew(arr)
        moments['kurtosis'] = stats.kurtosis(arr)
        moments['max'] = max(arr)
        moments['min'] = min(arr)
        return moments
