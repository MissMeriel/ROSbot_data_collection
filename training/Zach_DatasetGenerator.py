import numpy as np
import os, cv2, csv
# from DAVE2 import DAVE2Model
# from DAVE2pytorch import DAVE2PytorchModel
import kornia
from torchvision.transforms import ToPILImage
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

from data_augmentation.transformations import (
    add_shadow, time_of_day_transform_dusk, add_elastic_transform,
    add_blur_fn, color_jitter_fn, random_crop, adjust_brightness_fn,
    adjust_contrast_fn, adjust_saturation_fn, horizontal_flip,
    add_lens_distortion, add_noise
)


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
        # helper function to apply individual transformations to the image
        def custom_transform(image, transform_funcs):
            augmented_images = []
            for transform_func in transform_funcs:
                try:
                    # ensure image is in PIL format
                    if isinstance(image, torch.Tensor):
                        image = ToPILImage()(image)
                    augmented_image = transform_func(image, 0.5)  # Apply with 50% intensity
                    augmented_images.append(ToTensor()(augmented_image))
                except Exception as e:
                    print(f"Error applying {transform_func.__name__}: {e}")
            return augmented_images

        # helper function to apply composed transformations to the image
        def apply_composed_transformations(image, composed_transform_funcs):
            augmented_images = []
            for transform_func_list in composed_transform_funcs:
                try:
                    # ensure image is in PIL format
                    if isinstance(image, torch.Tensor):
                        image = ToPILImage()(image)
                    augmented_image = image
                    for transform_func in transform_func_list:
                        augmented_image = transform_func(augmented_image, 0.5)  # Apply with 50% intensity
                    augmented_images.append(ToTensor()(augmented_image))
                except Exception as e:
                    print(f"Error applying composed transformations {transform_func_list}: {e}")
            return augmented_images



        # Check if the sample is already in the cache
        if idx in self.cache:
            # Apply robustification if enabled
            if self.robustification:
                sample = self.cache[idx]
                y_steer = sample["angular_speed_z"]
                image = copy.deepcopy(sample["image name"])

                # Define the list of individual transformation functions
                # chatgpt
                transform_funcs = [
                    add_shadow, time_of_day_transform_dusk, add_elastic_transform,
                    add_blur_fn, color_jitter_fn, random_crop, adjust_brightness_fn,
                    adjust_contrast_fn, adjust_saturation_fn, horizontal_flip,
                    add_lens_distortion, add_noise
                ]

                # Define the list of composed transformation functions
                # chatgpt
                composed_transform_funcs = [
                    [add_shadow, time_of_day_transform_dusk],
                    [add_elastic_transform, add_blur_fn],
                    [color_jitter_fn, random_crop],
                    [adjust_brightness_fn, adjust_contrast_fn],
                    [adjust_saturation_fn, horizontal_flip],
                    [add_lens_distortion, add_noise]
                ]

                # Apply custom transformations
                transformed_images = custom_transform(image, transform_funcs)

                # Apply composed transformations
                composed_transformed_images = apply_composed_transformations(image, composed_transform_funcs)

                # Combine individual and composed transformations
                all_transformed_images = transformed_images + composed_transformed_images

                # Create the sample dictionary
                # create a list of augmented samples
                augmented_samples = []
                for img in all_transformed_images:
                    augmented_samples.append({
                        "image name": img,
                        "angular_speed_z": y_steer,
                        "linear_speed_x": sample["linear_speed_x"],
                        "lidar_ranges": sample['lidar_ranges'],
                        "all": torch.FloatTensor([y_steer, sample["linear_speed_x"]])
                    })

                return augmented_samples
            else:
                return self.cache[idx]

        # Load the image and resize it
        img_name = self.all_image_paths[idx]
        image = Image.open(img_name)
        image = image.resize(self.image_size)

        # Apply the initial transformation
        orig_image = self.transform(image)

        # Retrieve the corresponding steering and throttle values from the dataframe
        pathobj = Path(img_name)
        df = self.dfs_hashmap[f"{pathobj.parent}"]
        df_index = df.index[df['image name'] == img_name.name]
        orig_y_steer = df.loc[df_index, 'angular_speed_z'].item()
        y_throttle = df.loc[df_index, 'linear_speed_x'].item()

        # Convert y_steer to float tensor if necessary
        y_steer = torch.FloatTensor([orig_y_steer])
        y_throttle = torch.FloatTensor([y_throttle])


        # Convert lidar_ranges to list if it is a pandas Series
        lidar_ranges = df.loc[df_index, 'lidar_ranges'].tolist() if isinstance(df.loc[df_index, 'lidar_ranges'], pd.Series) else df.loc[df_index, 'lidar_ranges']


    # Define the list of individual transformation functions
        transform_funcs = [
            add_shadow, time_of_day_transform_dusk, add_elastic_transform,
            add_blur_fn, color_jitter_fn, random_crop, adjust_brightness_fn,
            adjust_contrast_fn, adjust_saturation_fn, horizontal_flip,
            add_lens_distortion, add_noise
        ]

        # Define the list of composed transformation functions
        composed_transform_funcs = [
            [add_shadow, time_of_day_transform_dusk],
            [add_elastic_transform, add_blur_fn],
            [color_jitter_fn, random_crop],
            [adjust_brightness_fn, adjust_contrast_fn],
            [adjust_saturation_fn, horizontal_flip],
            [add_lens_distortion, add_noise]
        ]

        # Apply custom transformations
        transformed_images = custom_transform(orig_image, transform_funcs)

        # Apply composed transformations
        composed_transformed_images = apply_composed_transformations(orig_image, composed_transform_funcs)

        # Combine individual and composed transformations
        all_transformed_images = transformed_images + composed_transformed_images

        # Visualize/logging to ensure appropriate transformations are applied
        if idx % 100 == 0:  # Visualize every 100th image
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(orig_image.permute(1, 2, 0))  # Original image
            ax[0].set_title(f'Original Image {idx}')
            ax[1].imshow(all_transformed_images[0].permute(1, 2, 0))  # First transformed image
            ax[1].set_title(f'Transformed Image {idx}')
            plt.show()
            print(f"Processed {len(transform_funcs) + len(composed_transform_funcs)} transformations for image {idx}")

        # create a list of augmented samples
        # create a list of augmented samples
        augmented_samples = []
        for img in all_transformed_images:
            augmented_samples.append({
                "image name": img,
                "angular_speed_z": y_steer,
                "linear_speed_x": y_throttle,
                "lidar_ranges": lidar_ranges,
                "all": torch.FloatTensor([y_steer, y_throttle])
            })

        orig_sample = {
            "image name": orig_image,
            "angular_speed_z": torch.FloatTensor([orig_y_steer]),
            "linear_speed_x": torch.FloatTensor([y_throttle]),
            "lidar_ranges": lidar_ranges,
            "all": torch.FloatTensor([orig_y_steer, y_throttle])
        }
def __getitem__(self, idx):
    # helper function to apply individual transformations to the image
    def custom_transform(image, transform_funcs):
        augmented_images = []
        for transform_func in transform_funcs:
            try:
                # ensure image is in PIL format
                if isinstance(image, torch.Tensor):
                    image = ToPILImage()(image)
                augmented_image = transform_func(image, 0.5)  # Apply with 50% intensity
                augmented_images.append(ToTensor()(augmented_image))
            except Exception as e:
                print(f"Error applying {transform_func.__name__}: {e}")
        return augmented_images

    # helper function to apply composed transformations to the image
    def apply_composed_transformations(image, composed_transform_funcs):
        augmented_images = []
        for transform_func_list in composed_transform_funcs:
            try:
                # ensure image is in PIL format
                if isinstance(image, torch.Tensor):
                    image = ToPILImage()(image)
                augmented_image = image
                for transform_func in transform_func_list:
                    augmented_image = transform_func(augmented_image, 0.5)  # Apply with 50% intensity
                augmented_images.append(ToTensor()(augmented_image))
            except Exception as e:
                print(f"Error applying composed transformations {transform_func_list}: {e}")
        return augmented_images



    # Check if the sample is already in the cache
    if idx in self.cache:
        # Apply robustification if enabled
        if self.robustification:
            sample = self.cache[idx]
            y_steer = sample["angular_speed_z"]
            image = copy.deepcopy(sample["image name"])

            # Define the list of individual transformation functions
            # chatgpt
            transform_funcs = [
                add_shadow, time_of_day_transform_dusk, add_elastic_transform,
                add_blur_fn, color_jitter_fn, random_crop, adjust_brightness_fn,
                adjust_contrast_fn, adjust_saturation_fn, horizontal_flip,
                add_lens_distortion, add_noise
            ]

            # Define the list of composed transformation functions
            # chatgpt
            composed_transform_funcs = [
                [add_shadow, time_of_day_transform_dusk],
                [add_elastic_transform, add_blur_fn],
                [color_jitter_fn, random_crop],
                [adjust_brightness_fn, adjust_contrast_fn],
                [adjust_saturation_fn, horizontal_flip],
                [add_lens_distortion, add_noise]
            ]

            # Apply custom transformations
            transformed_images = custom_transform(image, transform_funcs)

            # Apply composed transformations
            composed_transformed_images = apply_composed_transformations(image, composed_transform_funcs)

            # Combine individual and composed transformations
            all_transformed_images = transformed_images + composed_transformed_images

            # Create the sample dictionary
            # create a list of augmented samples
            augmented_samples = []
            for img in all_transformed_images:
                augmented_samples.append({
                    "image name": img,
                    "angular_speed_z": y_steer,
                    "linear_speed_x": sample["linear_speed_x"],
                    "lidar_ranges": sample['lidar_ranges'],
                    "all": torch.FloatTensor([y_steer, sample["linear_speed_x"]])
                })

            return augmented_samples
        else:
            return self.cache[idx]



    # Load the image and resize it
    img_name = self.all_image_paths[idx]
    image = Image.open(img_name)
    if self.transform:
        image = self.transform(image)

    # Apply the initial transformation
    orig_image = self.transform(image)



    # Retrieve the corresponding steering and throttle values from the dataframe
    pathobj = Path(img_name)
    df = self.dfs_hashmap[f"{pathobj.parent}"]
    df_index = df.index[df['image name'] == img_name.name]
    orig_y_steer = df.loc[df_index, 'angular_speed_z'].item()
    y_throttle = df.loc[df_index, 'linear_speed_x'].item()

    # Convert y_steer to float tensor if necessary
    y_steer = torch.FloatTensor([orig_y_steer])
    y_throttle = torch.FloatTensor([y_throttle])


    # Convert lidar_ranges to list if it is a pandas Series
    lidar_ranges = df.loc[df_index, 'lidar_ranges'].tolist() if isinstance(df.loc[df_index, 'lidar_ranges'], pd.Series) else df.loc[df_index, 'lidar_ranges']


    # Define the list of individual transformation functions
    transform_funcs = [
        add_shadow, time_of_day_transform_dusk, add_elastic_transform,
        add_blur_fn, color_jitter_fn, random_crop, adjust_brightness_fn,
        adjust_contrast_fn, adjust_saturation_fn, horizontal_flip,
        add_lens_distortion, add_noise
    ]

    # Define the list of composed transformation functions
    composed_transform_funcs = [
        [add_shadow, time_of_day_transform_dusk],
        [add_elastic_transform, add_blur_fn],
        [color_jitter_fn, random_crop],
        [adjust_brightness_fn, adjust_contrast_fn],
        [adjust_saturation_fn, horizontal_flip],
        [add_lens_distortion, add_noise]
    ]

    # Apply custom transformations
    transformed_images = custom_transform(orig_image, transform_funcs)

    # Apply composed transformations
    composed_transformed_images = apply_composed_transformations(orig_image, composed_transform_funcs)

    # Combine individual and composed transformations
    all_transformed_images = transformed_images + composed_transformed_images

    # Visualize/logging to ensure appropriate transformations are applied
    if idx % 100 == 0:  # Visualize every 100th image
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(orig_image.permute(1, 2, 0))  # Original image
        ax[0].set_title(f'Original Image {idx}')
        ax[1].imshow(all_transformed_images[0].permute(1, 2, 0))  # First transformed image
        ax[1].set_title(f'Transformed Image {idx}')
        plt.show()
        print(f"Processed {len(transform_funcs) + len(composed_transform_funcs)} transformations for image {idx}")

    # create a list of augmented samples
    # create a list of augmented samples
    augmented_samples = []
    for img in all_transformed_images:
        augmented_samples.append({
            "image name": img,
            "angular_speed_z": y_steer,
            "linear_speed_x": y_throttle,
            "lidar_ranges": lidar_ranges,
            "all": torch.FloatTensor([y_steer, y_throttle])
        })

    orig_sample = {
        "image name": orig_image,
        "angular_speed_z": torch.FloatTensor([orig_y_steer]),
        "linear_speed_x": torch.FloatTensor([y_throttle]),
        "lidar_ranges": lidar_ranges,
        "all": torch.FloatTensor([orig_y_steer, y_throttle])
    }

    self.cache[idx] = orig_sample

    return augmented_samples
        self.cache[idx] = orig_sample

        return augmented_samples






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
