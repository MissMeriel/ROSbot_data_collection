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

def add_shadow(image):
    width, height = image.size
    x1, y1 = width * random.uniform(0, 1), 0
    x2, y2 = width * random.uniform(0, 1), height
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for i in range(width):
        for j in range(height):
            a = (i - x1) * (y2 - y1) - (x2 - x1) * (j - y1)
            if a > 0:
                shadow.putpixel((i, j), (0, 0, 0, 127))
    combined = Image.alpha_composite(image.convert('RGBA'), shadow)
    return combined.convert('RGB')

def add_fog(image):
    width, height = image.size
    fog = Image.new('RGBA', (width, height), (255, 255, 255, 50))
    combined = Image.alpha_composite(image.convert('RGBA'), fog)
    return combined.convert('RGB')

def time_of_day_transform_dusk(image):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(0.5)

def time_of_day_transform_dawn(image):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1.5)

def add_elastic_transform(image):
    image = np.array(image)
    transform = A.ElasticTransform(p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_lens_distortion(image):
    image = np.array(image)
    transform = A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_noise(image, noise_level):
    array = np.array(image)
    noise = np.random.normal(0, noise_level * 255, array.shape)
    array = np.clip(array + noise, 0, 255)
    return Image.fromarray(array.astype('uint8'))

def add_blur_fn(img, blur_level):
    kernel_size = int(np.ceil(blur_level * 8))  # Kernel size from 1 to 8
    if kernel_size % 2 == 0:  # Kernel size must be odd
        kernel_size += 1
    return transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 5))(img)

def adjust_brightness_fn(img, brightness_level):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_level)

def adjust_contrast_fn(img, contrast_level):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(contrast_level)

def adjust_saturation_fn(img, saturation_level):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(saturation_level)

def adjust_hue_fn(img, hue_level):
    return transforms.functional.adjust_hue(img, hue_level)

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
    "shadow": lambda img: add_shadow(img),
    "fog": lambda img: add_fog(img),
    "time_of_day_dusk": lambda img: time_of_day_transform_dusk(img),
    "time_of_day_dawn": lambda img: time_of_day_transform_dawn(img),
    "elastic_transform": lambda img: add_elastic_transform(img),
    "lens_distortion": lambda img, level: add_lens_distortion(img),
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
        transforms.Lambda(lambda img: add_lens_distortion(img)),
        transforms.Lambda(lambda img: add_noise(img, level)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "lens_distortion_noise": lambda img, level: transforms.Compose([
        transforms.Lambda(lambda img: add_lens_distortion(img)),
        transforms.Lambda(lambda img: add_noise(img, level)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "only_lens_distortion": lambda img: transforms.Compose([
        transforms.Lambda(lambda img: add_lens_distortion(img)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "dusk_shadow_rotation": lambda img: transforms.Compose([
        transforms.Lambda(lambda img: time_of_day_transform_dusk(img)),
        transforms.Lambda(lambda img: add_shadow(img)),
        individual_transforms["random_rotation"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "dawn_shadow_rotation": lambda img: transforms.Compose([
        transforms.Lambda(lambda img: time_of_day_transform_dawn(img)),
        transforms.Lambda(lambda img: add_shadow(img)),
        individual_transforms["random_rotation"](img),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
}

# Combine both dictionaries
all_transforms_dict = {**individual_transforms, **composed_transforms}

def augment_and_save_image(args):
    image_path, output_dir, transform_name, level, row = args
    try:
        image = Image.open(image_path)
        if level is not None:
            augmented_image = all_transforms_dict[transform_name](image, level)
        else:
            augmented_image = all_transforms_dict[transform_name](image)
        save_path = output_dir / f"{image_path.stem}_{transform_name}_{level}.jpg"
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
            for level in range(1, 101):  # 1% to 100% in 1% increments
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

def process_parent_dir(parentdir, level, img_filename_key="image name"):
    if level == 'rosbotxl_data':
        collection_dirs = [p for p in Path(parentdir).iterdir() if p.is_dir()]
    elif level == 'collection':
        collection_dirs = [Path(parentdir)]

    for collection_dir in collection_dirs:
        process_collection_dir(collection_dir, img_filename_key)

if __name__ == '__main__':
    process_parent_dir(args.parentdir, args.level, args.img_filename_key)
