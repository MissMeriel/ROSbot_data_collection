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

def add_noise(image):
    array = np.array(image)
    noise = np.random.normal(0, 25, array.shape)
    array = np.clip(array + noise, 0, 255)
    return Image.fromarray(array.astype('uint8'))

def add_blur_fn(img):
    return transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)

def add_lens_distortion_fn(img):
    return add_lens_distortion(img)

def add_noise_fn(img):
    return add_noise(img)

def add_shadow_fn(img):
    return add_shadow(img)

def add_fog_fn(img):
    return add_fog(img)

def time_of_day_transform_dusk_fn(img):
    return time_of_day_transform_dusk(img)

def time_of_day_transform_dawn_fn(img):
    return time_of_day_transform_dawn(img)

def add_elastic_transform_fn(img):
    return add_elastic_transform(img)

# Define individual transformations
individual_transforms = {
    "blur": transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    "color_jitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.1),
    "random_rotation": transforms.RandomRotation(30),
    "horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),
    "random_crop": transforms.RandomResizedCrop(224),
    "brightness": transforms.ColorJitter(brightness=0.5),
    "contrast": transforms.ColorJitter(contrast=0.5),
    "scaling_zooming": transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    "shadow": transforms.Lambda(add_shadow_fn),
    "fog": transforms.Lambda(add_fog_fn),
    "time_of_day_dusk": transforms.Lambda(time_of_day_transform_dusk_fn),
    "time_of_day_dawn": transforms.Lambda(time_of_day_transform_dawn_fn),
    "elastic_transform": transforms.Lambda(add_elastic_transform_fn),
    "lens_distortion": transforms.Lambda(add_lens_distortion_fn),
    "noise": transforms.Lambda(add_noise_fn)
}

# Define composed transformations
composed_transforms = {
    "blur_color_jitter_rotation": transforms.Compose([
        individual_transforms["blur"],
        individual_transforms["color_jitter"],
        individual_transforms["random_rotation"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "horizontal_flip_blur": transforms.Compose([
        individual_transforms["horizontal_flip"],
        individual_transforms["blur"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "brightness_contrast_rotation": transforms.Compose([
        individual_transforms["brightness"],
        individual_transforms["contrast"],
        individual_transforms["random_rotation"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "horizontal_flip_crop": transforms.Compose([
        individual_transforms["horizontal_flip"],
        individual_transforms["random_crop"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "brightness_scaling_zooming": transforms.Compose([
        individual_transforms["brightness"],
        individual_transforms["scaling_zooming"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "brightness_blur": transforms.Compose([
        individual_transforms["brightness"],
        individual_transforms["blur"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "blur_lens_distortion_noise": transforms.Compose([
        transforms.Lambda(add_blur_fn),
        transforms.Lambda(add_lens_distortion_fn),
        transforms.Lambda(add_noise_fn),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "lens_distortion_noise": transforms.Compose([
        transforms.Lambda(add_lens_distortion_fn),
        transforms.Lambda(add_noise_fn),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "only_lens_distortion": transforms.Compose([
        transforms.Lambda(add_lens_distortion_fn),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "dusk_shadow_rotation": transforms.Compose([
        transforms.Lambda(time_of_day_transform_dusk_fn),
        transforms.Lambda(add_shadow_fn),
        individual_transforms["random_rotation"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ]),
    "dawn_shadow_rotation": transforms.Compose([
        transforms.Lambda(time_of_day_transform_dawn_fn),
        transforms.Lambda(add_shadow_fn),
        individual_transforms["random_rotation"],
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
}

# Combine both dictionaries
all_transforms_dict = {**individual_transforms, **composed_transforms}

def augment_and_save_image(args):
    image_path, output_dir, transform, transform_name, row = args
    try:
        image = Image.open(image_path)
        augmented_image = transform(image)
        save_path = output_dir / image_path.name
        augmented_image.save(save_path)

        # Update the CSV file row if the transformation is horizontal flip
        if 'horizontal_flip' in transform_name:
            row['angular_speed_z'] = -row['angular_speed_z']
        
        # Save the new image path and row data to the CSV file
        new_image_name = str(output_dir / image_path.name)
        row['image name'] = new_image_name
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
            output_dir = main_output_dir / transform_name
            output_dir.mkdir(parents=True, exist_ok=True)
            row = df[df[img_filename_key] == pp.name].iloc[0].copy()
            tasks.append((pp, output_dir, transform, transform_name, row))

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
