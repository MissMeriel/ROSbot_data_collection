import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
import random
import albumentations as A
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Parse command line args into 'args' variable
parser = argparse.ArgumentParser()
parser.add_argument("--parentdir", type=str, default='/home/husarion/media/usb/rosbotxl_data')
parser.add_argument("--img_filename_key", type=str, default="image name")
parser.add_argument("--level", type=str, choices=['rosbotxl_data', 'collection'], default='rosbotxl_data',
                    help="Specify the directory level to process: 'rosbotxl_data' for the whole dataset or 'collection' for a single collection.")
args = parser.parse_args()
print("args:" + str(args))

def add_shadow(image, level=0.5):
    """
    Adds a shadow effect to the given image.

    Args:
    - image (PIL.Image): The input image to which the shadow effect will be applied.
    - level (float): The intensity level of the shadow effect (0.01 to 0.9).

    Returns:
    - PIL.Image: The image with the shadow effect applied.
    """
    level = min(max(level, 0.01), 0.9)
    width, height = image.size
    x1, y1 = width * random.uniform(0, 1), 0
    x2, y2 = width * random.uniform(0, 1), height
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for i in range(width):
        for j in range(height):
            a = (i - x1) * (y2 - y1) - (x2 - x1) * (j - y1)
            if a > 0:
                shadow.putpixel((i, j), (0, 0, 0, int(127 * level)))
    combined = Image.alpha_composite(image.convert('RGBA'), shadow)
    return combined.convert('RGB')

def add_fog(image, level=0.5):
    """
    Adds a fog effect to the given image.

    Args:
    - image (PIL.Image): The input image to which the fog effect will be applied.
    - level (float): The intensity level of the fog effect (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with the fog effect applied.
    """
    level = min(max(level, 0.01), 1.0)
    width, height = image.size
    fog = Image.new('RGBA', (width, height), (255, 255, 255, int(255 * level)))
    combined = Image.alpha_composite(image.convert('RGBA'), fog)
    return combined.convert('RGB')

def time_of_day_transform_dusk(image, level=0.5):
    """
    Applies a dusk time-of-day effect to the given image by reducing the color saturation.

    Args:
    - image (PIL.Image): The input image to which the dusk effect will be applied.
    - level (float): The intensity level of the dusk effect (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with the dusk effect applied.
    """
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 - level)

def time_of_day_transform_dawn(image, level=0.5):
    """
    Applies a dawn time-of-day effect to the given image by increasing the color saturation.

    Args:
    - image (PIL.Image): The input image to which the dawn effect will be applied.
    - level (float): The intensity level of the dawn effect (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with the dawn effect applied.
    """
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 + level)

def add_elastic_transform(image, level=0.5):
    """
    Applies an elastic transformation to the given image.

    Args:
    - image (PIL.Image): The input image to which the elastic transformation will be applied.
    - level (float): The intensity level of the elastic transformation (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with the elastic transformation applied.
    """
    level = min(max(level, 0.01), 1.0)
    image = np.array(image)
    transform = A.ElasticTransform(alpha=level * 100, sigma=level * 10, alpha_affine=level * 10, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_lens_distortion(image, level=0.5):
    """
    Applies a lens distortion effect to the given image.

    Args:
    - image (PIL.Image): The input image to which the lens distortion effect will be applied.
    - level (float): The intensity level of the lens distortion effect (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with the lens distortion effect applied.
    """
    level = min(max(level, 0.01), 1.0)
    image = np.array(image)
    transform = A.OpticalDistortion(distort_limit=0.2 * level, shift_limit=0.2 * level, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_noise(image, level=0.5):
    """
    Adds Gaussian noise to the given image.

    Args:
    - image (PIL.Image): The input image to which the noise will be added.
    - level (float): The standard deviation of the Gaussian noise to be added (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with Gaussian noise added.
    """
    level = min(max(level, 0.01), 1.0)
    array = np.array(image)
    noise = np.random.normal(0, level * 255, array.shape)
    array = np.clip(array + noise, 0, 255)
    return Image.fromarray(array.astype('uint8'))

def add_blur_fn(img, level=0.5):
    """
    Adds a Gaussian blur effect to the given image.

    Args:
    - img (PIL.Image): The input image to which the blur effect will be applied.
    - level (float): The level of blur to apply, controlling the kernel size (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with the blur effect applied.
    """
    level = min(max(level, 0.01), 1.0)
    kernel_size = int(np.ceil(level * 8))  # Kernel size from 1 to 8
    if kernel_size % 2 == 0:  # Kernel size must be odd
        kernel_size += 1
    return transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 5))(img)

def adjust_brightness_fn(img, level=0.5):
    """
    Adjusts the brightness of the given image.

    Args:
    - img (PIL.Image): The input image whose brightness will be adjusted.
    - level (float): The factor by which to adjust the brightness (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with adjusted brightness.
    """
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(level)

def adjust_contrast_fn(img, level=0.5):
    """
    Adjusts the contrast of the given image.

    Args:
    - img (PIL.Image): The input image whose contrast will be adjusted.
    - level (float): The factor by which to adjust the contrast (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with adjusted contrast.
    """
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(level)

def adjust_saturation_fn(img, level=0.5):
    """
    Adjusts the saturation of the given image.

    Args:
    - img (PIL.Image): The input image whose saturation will be adjusted.
    - level (float): The factor by which to adjust the saturation (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with adjusted saturation.
    """
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(level)

def adjust_hue_fn(img, level=0.5):
    """
    Adjusts the hue of the given image.

    Args:
    - img (PIL.Image): The input image whose hue will be adjusted.
    - level (float): The factor by which to adjust the hue (0.01 to 1.0).

    Returns:
    - PIL.Image: The image with adjusted hue.
    """
    level = min(max(level, 0.01), 1.0)
    return transforms.functional.adjust_hue(img, level)

# Define individual transformations with adjustable parameters
individual_transforms = {
    "blur": lambda img, level: add_blur_fn(img, level),
    "color_jitter": lambda img, level: transforms.ColorJitter(
        brightness=level,
        contrast=level,
        hue=level * 0.5,  # Hue adjustment ranges from -0.5 to 0.5
        saturation=level
    )(img),
    "random_rotation": transforms.RandomRotation(30),
    "horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),
    "random_crop": transforms.RandomResizedCrop(224),
    "brightness": lambda img, level: adjust_brightness_fn(img, level),
    "contrast": lambda img, level: adjust_contrast_fn(img, level),
    "scaling_zooming": transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    "shadow": lambda img, level: add_shadow(img, level),
    "fog": lambda img, level: add_fog(img, level),
    "time_of_day_dusk": lambda img, level: time_of_day_transform_dusk(img, level),
    "time_of_day_dawn": lambda img, level: time_of_day_transform_dawn(img, level),
    "elastic_transform": lambda img, level: add_elastic_transform(img, level),
    "lens_distortion": lambda img, level: add_lens_distortion(img, level),
    "noise": lambda img, level: add_noise(img, level)
}

# Define composed transformations
composed_transforms = {
    "blur_color_jitter_rotation": lambda img, level: transforms.Compose([
        individual_transforms["blur"](img, level),
        individual_transforms["color_jitter"](img, level),
        individual_transforms["random_rotation"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "horizontal_flip_blur": lambda img, level: transforms.Compose([
        individual_transforms["horizontal_flip"](img),
        individual_transforms["blur"](img, level),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "brightness_contrast_rotation": lambda img, level: transforms.Compose([
        individual_transforms["brightness"](img, level),
        individual_transforms["contrast"](img, level),
        individual_transforms["random_rotation"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "horizontal_flip_crop": lambda img: transforms.Compose([
        individual_transforms["horizontal_flip"](img),
        individual_transforms["random_crop"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "brightness_scaling_zooming": lambda img, level: transforms.Compose([
        individual_transforms["brightness"](img, level),
        individual_transforms["scaling_zooming"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "brightness_blur": lambda img, level: transforms.Compose([
        individual_transforms["brightness"](img, level),
        individual_transforms["blur"](img, level),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "blur_lens_distortion_noise": lambda img, level: transforms.Compose([
        transforms.Lambda(lambda img: add_blur_fn(img, level)),
        transforms.Lambda(lambda img: add_lens_distortion(img, level)),
        transforms.Lambda(lambda img: add_noise(img, level)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "lens_distortion_noise": lambda img, level: transforms.Compose([
        transforms.Lambda(lambda img: add_lens_distortion(img, level)),
        transforms.Lambda(lambda img: add_noise(img, level)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "only_lens_distortion": lambda img: transforms.Compose([
        transforms.Lambda(lambda img: add_lens_distortion(img, 0.5)),  # Default level for this composed transform
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "dusk_shadow_rotation": lambda img, level: transforms.Compose([
        transforms.Lambda(lambda img: time_of_day_transform_dusk(img, level)),
        transforms.Lambda(lambda img: add_shadow(img, level)),
        individual_transforms["random_rotation"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "dawn_shadow_rotation": lambda img, level: transforms.Compose([
        transforms.Lambda(lambda img: time_of_day_transform_dawn(img, level)),
        transforms.Lambda(lambda img: add_shadow(img, level)),
        individual_transforms["random_rotation"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
}

# Combine both dictionaries
all_transforms_dict = {**individual_transforms, **composed_transforms}

def augment_and_save_image(args):
    """
    Applies a specified transformation to an image and saves the result.

    Args:
    - args (tuple): Contains the following elements:
        - image_path (Path): Path to the input image.
        - output_dir (Path): Directory where the augmented image will be saved.
        - transform_name (str): Name of the transformation to apply.
        - level (float): Level or intensity of the transformation.
        - row (pd.Series): Row from the CSV file corresponding to the image.

    Returns:
    - pd.Series: Updated row with the new image name if the transformation is horizontal flip.
    """
    image_path, output_dir, transform_name, level, row = args
    try:
        image = Image.open(image_path)
        if level is not None:
            augmented_image = all_transforms_dict[transform_name](image, level)
        else:
            augmented_image = all_transforms_dict[transform_name](image)
        save_path = output_dir / f"{image_path.stem}_{transform_name}_{int(level * 100)}.jpg"
        augmented_image.save(save_path)

        # Update the CSV file row if the transformation is horizontal flip
        if 'horizontal_flip' in transform_name:
            row['angular_speed_z'] = -row['angular_speed_z']

        # Save the new image path and row data to the CSV file
        row['image name'] = save_path.name
        return row
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return None

def process_collection_dir(collection_dir, img_filename_key="image name"):
    """
    Processes a directory containing a collection of images, applying augmentations and saving the results.

    Args:
    - collection_dir (Path): Path to the directory containing the image collection.
    - img_filename_key (str): Key in the CSV file that corresponds to image filenames.

    Returns:
    - None
    """
    main_output_dir = collection_dir / "augmented_data"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    data_csv_path = collection_dir / "data.csv"
    augmented_data_csv_path = main_output_dir / "augmented_data.csv"
    df = pd.read_csv(data_csv_path)
    augmented_df = pd.DataFrame(columns=df.columns)

    image_paths = [pp for pp in Path(collection_dir).iterdir() if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
    tasks = []

    for pp in image_paths:
        for transform_name, transform in all_transforms_dict.items():
            max_level = 90 if transform_name == "shadow" else 100
            for level in range(1, max_level + 1):  # 1% to 100% in 1% increments, 1% to 90% for shadow
                level_value = level / 100
                output_dir = main_output_dir / transform_name
                output_dir.mkdir(parents=True, exist_ok=True)
                row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                tasks.append((pp, output_dir, transform_name, level_value, row))

    # Use multiprocessing Pool to process images in parallel
    num_workers = cpu_count()
    start_time = time.time()
    with Pool(num_workers) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(augment_and_save_image, tasks), total=len(tasks)):
            if result is not None:
                results.append(result)

    end_time = time.time()
    augmented_df = pd.DataFrame(results)
    augmented_df.to_csv(augmented_data_csv_path, index=False)

    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.2f} seconds")


#ignore
def process_parent_dir(parentdir, level, img_filename_key="image name"):
    """
    Processes the parent directory containing multiple collections of images.

    Args:
    - parentdir (str): Path to the parent directory.
    - level (str): The level of directory processing ('rosbotxl_data' or 'collection').
    - img_filename_key (str): Key in the CSV file that corresponds to image filenames.

    Returns:
    - None
    """
    if level == 'rosbotxl_data':
        collection_dirs = [p for p in Path(parentdir).iterdir() if p.is_dir()]
    elif level == 'collection':
        collection_dirs = [Path(parentdir)]

    for collection_dir in collection_dirs:
        process_collection_dir(collection_dir, img_filename_key)

if __name__ == '__main__':
    process_parent_dir(args.parentdir, args.level, args.img_filename_key)
