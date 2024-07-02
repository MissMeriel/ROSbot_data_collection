# processing.py

import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import traceback
import transformations

# Combine both dictionaries
all_transforms_dict = transformations.all_transforms_dict

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
        # Open the image
        image = Image.open(image_path)
        # Apply transformation with the provided level
        if level is not None:
            augmented_image = all_transforms_dict[transform_name](image, level)
        else:
            augmented_image = all_transforms_dict[transform_name](image)
        # Save the transformed image to the specified path
        save_path = output_dir / f"{image_path.stem}_{transform_name}_{int(level * 100)}.jpg"
        augmented_image.save(save_path)
        # Update the CSV file row if the transformation is horizontal flip
        if 'horizontal_flip' in transform_name:
            row['angular_speed_z'] = -row['angular_speed_z']
            if isinstance(row['lidar_ranges'], str):
                lidar_data = np.array([float(x) for x in row['lidar_ranges'].split()])
            else:
                lidar_data = np.array([float(row['lidar_ranges'])])
            flipped_lidar_data = np.flip(lidar_data).tolist()
            row['lidar_ranges'] = ' '.join(map(str, flipped_lidar_data))
        row['image name'] = save_path.name
        return row
    except Exception as e:
        # Print error message and traceback if an exception occurs
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
    return None

def augment_and_save_composed_image(args):
    """
    Applies a composed transformation to an image and saves the result.

    Args:
    - args (tuple): Contains the following elements:
        - image_path (Path): Path to the input image.
        - output_dir (Path): Directory where the augmented image will be saved.
        - transform_names (list): Names of the transformations to apply in sequence.
        - level (float): Level or intensity of the transformation.
        - row (pd.Series): Row from the CSV file corresponding to the image.

    Returns:
    - pd.Series: Updated row with the new image name if the transformation is horizontal flip.
    """
    image_path, output_dir, transform_names, level, row = args
    try:
        # Open the image
        image = Image.open(image_path)
        # Compose the transformations
        transformations = [all_transforms_dict[name] for name in transform_names]
        composed_transform = transformations.compose_transformations(transformations)
        # Apply the composed transformation with the provided level
        augmented_image = composed_transform(image, level)
        # Create a name for the composed transformation
        composed_name = "_".join(transform_names)
        # Save the transformed image to the specified path
        save_path = output_dir / f"{image_path.stem}_{composed_name}_{int(level * 100)}.jpg"
        augmented_image.save(save_path)
        # Update the CSV file row if the transformation includes horizontal flip
        if 'horizontal_flip' in transform_names:
            row['angular_speed_z'] = -row['angular_speed_z']
            lidar_data = np.array([float(x) for x in row['lidar_ranges'].split()])
            flipped_lidar_data = np.flip(lidar_data).tolist()
            row['lidar_ranges'] = ' '.join(map(str, flipped_lidar_data))
        row['image name'] = save_path.name
        return row
    except Exception as e:
        # Print error message and traceback if an exception occurs
        print(f"Error processing {image_path}: {e}")
        traceback.print_exc()
    return None

def process_collection_dir(collection_dir, img_filename_key="image name", transformations_list=None, composed_transformations_list=None, specified_images=None):
    main_output_dir = collection_dir / "augmented_data"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    data_csv_path = collection_dir / "data.csv"
    augmented_data_csv_path = main_output_dir / "augmented_data.csv"
    df = pd.read_csv(data_csv_path)
    augmented_df = pd.DataFrame(columns=df.columns)

    if specified_images:
        image_paths = [Path(image) for image in specified_images]
    else:
        image_paths = [pp for pp in Path(collection_dir).iterdir() if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]

    tasks = []

    if transformations_list:
        for pp in image_paths:
            for transform_name in transformations_list:
                if transform_name != 'horizontal_flip':  # Exclude horizontal_flip
                    for level in range(5, 80, 5):  # 5% to 75% in 5% increments
                        level_value = level / 100
                        output_dir = main_output_dir / transform_name
                        output_dir.mkdir(parents=True, exist_ok=True)
                        row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                        tasks.append((pp, output_dir, transform_name, level_value, row))
                else:
                    output_dir = main_output_dir / transform_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                    tasks.append((pp, output_dir, transform_name, 2, row))

    if composed_transformations_list:
        for pp in image_paths:
            for transform_names in composed_transformations_list:
                for level in range(5, 80, 5):  # 5% to 75% in 5% increments
                    level_value = level / 100
                    composed_name = "_".join(transform_names)
                    output_dir = main_output_dir / composed_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                    tasks.append((pp, output_dir, transform_names, level_value, row))

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

def process_parent_dir(parentdir, level, img_filename_key="image name", transformations_list=None, composed_transformations_list=None, specified_images=None):
    if level == 'rosbotxl_data':
        collection_dirs = [p for p in Path(parentdir).iterdir() if p.is_dir()]
    elif level == 'collection':
        collection_dirs = [Path(parentdir)]

    for collection_dir in collection_dirs:
        process_collection_dir(collection_dir, img_filename_key, transformations_list, composed_transformations_list, specified_images)

if __name__ == '__main__':
    process_parent_dir(args.parentdir, args.level, args.img_filename_key)
