# processing.py

import traceback
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import deque
from transformations import all_transforms_dict, compose_transformations

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
        composed_transform = compose_transformations(transformations)
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

def process_collection_dir(collection_dir, img_filename_key="image name", transformations=None, composed_transforms=None, specified_images=None):
    """
    Processes a directory containing a collection of images, applying augmentations and saving the results.

    Args:
    - collection_dir (Path): Path to the directory containing the image collection.
    - img_filename_key (str): Key in the CSV file that corresponds to image filenames.
    - transformations (list): List of individual transformations to apply.
    - composed_transforms (list): List of composed transformations to apply.
    - specified_images (list): List of specific image paths to process.

    Returns:
    - None
    """
    # Create main output directory for augmented images
    main_output_dir = collection_dir / "augmented_data"
    main_output_dir.mkdir(parents=True, exist_ok=True)
    # Paths to the CSV files
    data_csv_path = collection_dir / "data.csv"
    augmented_data_csv_path = main_output_dir / "augmented_data.csv"
    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_csv_path)
    augmented_df = pd.DataFrame(columns=df.columns)
    # Get paths to all images in the collection directory
    image_paths = [pp for pp in Path(collection_dir).iterdir() if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
    # If specific images are specified, use only those images
    if specified_images:
        image_paths = [Path(img) for img in specified_images]
    # Initialize deque for tasks
    tasks = deque()
    # Loop through each image path
    for pp in image_paths:
        # Apply specified individual transformations
        if transformations:
            for transform_name in transformations:
                # Define the maximum possible intensity
                max_level = 80
                # For each image, create a new image where the intensity is increased by 5% per step
                for level in range(5, max_level + 5, 5):
                    level_value = level / 100
                    # Create output directory for each transformation
                    output_dir = main_output_dir / transform_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Select the row for each image from the DataFrame
                    row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                    # Add the task tuple to the tasks deque
                    tasks.append((pp, output_dir, transform_name, level_value, row))
        # Apply composed transformations if specified
        if composed_transforms:
            for composed_transform in composed_transforms:
                composed_name = "_".join(composed_transform)
                for level in range(5, max_level + 5, 5):
                    level_value = level / 100
                    output_dir = main_output_dir / composed_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                    tasks.append((pp, output_dir, composed_transform, level_value, row))
    # Use multiprocessing Pool to process images in parallel
    num_workers = cpu_count()
    start_time = time.time()
    # Create a pool of workers
    with Pool(num_workers) as pool:
        results = []
        # Process the tasks in parallel and collect the results
        for result in tqdm(pool.imap_unordered(augment_and_save_image if len(result[2]) == 1 else augment_and_save_composed_image, tasks), total=len(tasks)):
            # Append valid (non-null) results to the results list
            if result is not None:
                results.append(result)
    # Record end time, write results to the new CSV file, and print the runtime
    end_time = time.time()
    augmented_df = pd.DataFrame(results)
    augmented_df.to_csv(augmented_data_csv_path, index=False)
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.2f} seconds")

def process_parent_dir(parentdir, level, img_filename_key="image name", transformations=None, composed_transforms=None, specified_images=None):
    """
    Processes the parent directory containing multiple collections of images.

    Args:
    - parentdir (str): Path to the parent directory.
    - level (str): The level of directory processing ('rosbotxl_data' or 'collection').
    - img_filename_key (str): Key in the CSV file that corresponds to image filenames.
    - transformations (list): List of individual transformations to apply.
    - composed_transforms (list): List of composed transformations to apply.
    - specified_images (list): List of specific image paths to process.

    Returns:
    - None
    """
    # Determine the collection directories based on the specified level
    if level == 'rosbotxl_data':
        # If processing the entire dataset, get all subdirectories
        collection_dirs = [p for p in Path(parentdir).iterdir() if p.is_dir()]
    elif level == 'collection':
        # If processing a single collection, use the parent directory itself
        collection_dirs = [Path(parentdir)]
    # Process each collection directory
    for collection_dir in collection_dirs:
        process_collection_dir(collection_dir, img_filename_key, transformations, composed_transforms, specified_images)
