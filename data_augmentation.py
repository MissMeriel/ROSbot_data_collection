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
    level = min(max(level, 0.01), 1.0)
    width, height = image.size
    fog = Image.new('RGBA', (width, height), (255, 255, 255, int(255 * level)))
    combined = Image.alpha_composite(image.convert('RGBA'), fog)
    return combined.convert('RGB')

def time_of_day_transform_dusk(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 - level)

def time_of_day_transform_dawn(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 + level)

def add_elastic_transform(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    image = np.array(image)
    transform = A.ElasticTransform(alpha=level * 100, sigma=level * 10, alpha_affine=level * 10, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_lens_distortion(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    image = np.array(image)
    transform = A.OpticalDistortion(distort_limit=0.2 * level, shift_limit=0.2 * level, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_noise(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    array = np.array(image)
    noise = np.random.normal(0, level * 255, array.shape)
    array = np.clip(array + noise, 0, 255)
    return Image.fromarray(array.astype('uint8'))

def add_blur_fn(img, level=0.5):
    level = min(max(level, 0.01), 1.0)
    kernel_size = int(np.ceil(level * 8))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 5))(img)

def adjust_brightness_fn(img, level=0.5):
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(level)

def adjust_contrast_fn(img, level=0.5):
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(level)

def adjust_saturation_fn(img, level=0.5):
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(level)

def adjust_hue_fn(img, level=0.5):
    level = min(max(level, 0.01), 1.0)
    return transforms.functional.adjust_hue(img, level)

def horizontal_flip(image, level=1.0):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def random_crop(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    crop_size = int(level * min(image.size))
    return transforms.RandomCrop(crop_size)(image)

def compose_transformations(transformations):
    """
    Composes a sequence of transformations into a single transformation function.

    Args:
    - transformations (list): A list of tuples where each tuple contains a transformation function and its corresponding level.

    Returns:
    - function: A function that applies the composed transformations to an image.
    """
    def composed_transformation(image):
        for transform, level in transformations:
            image = transform(image, level) if level is not None else transform(image)
        return image
    return composed_transformation


composed_transforms = {
    "dusk_fog": compose_transformations([(time_of_day_transform_dusk, 0.5), (add_fog, 0.5)]),
    "dawn_shadow": compose_transformations([(time_of_day_transform_dawn, 0.5), (add_shadow, 0.5)]),
    "elastic_noise": compose_transformations([(add_elastic_transform, 0.5), (add_noise, 0.5)]),
    "blur_brightness_flip": compose_transformations([(add_blur_fn, 0.5), (adjust_brightness_fn, 0.5), (horizontal_flip, None)]),
    "crop_contrast": compose_transformations([(random_crop, 0.5), (adjust_contrast_fn, 0.5)])
}

# Define both individual and composed transformations
all_transforms_dict = {**individual_transforms_with_level, **individual_transformations_without_level, **composed_transforms}



# Define individual and composed transformations
individual_transforms_with_level = {
    "blur": add_blur_fn,
    "color_jitter": transforms.ColorJitter,
    "random_crop": random_crop,
    "brightness": adjust_brightness_fn,
    "contrast": adjust_contrast_fn,
    "shadow": add_shadow,
    "fog": add_fog,
    "time_of_day_dusk": time_of_day_transform_dusk,
    "time_of_day_dawn": time_of_day_transform_dawn,
    "elastic_transform": add_elastic_transform,
    "lens_distortion": add_lens_distortion,
    "noise": add_noise
}

individual_transformations_without_level = {
	"horizontal_flip": horizontal_flip,
}


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
    
    #unpack args
    image_path, output_dir, transform_name, level, row = args
    
    
    try:
    
    	#open the image
        image = Image.open(image_path)
        
        #use intensity level if provided
        
        if level is not None:
            augmented_image = all_transforms_dict[transform_name](image, level)
        else:
            augmented_image = all_transforms_dict[transform_name](image)
            
        #save the path and images
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
        traceback.print_exc()
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
    
    #intiialization
    main_output_dir = collection_dir / "augmented_data"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    data_csv_path = collection_dir / "data.csv"
    augmented_data_csv_path = main_output_dir / "augmented_data.csv"
    df = pd.read_csv(data_csv_path)
    augmented_df = pd.DataFrame(columns=df.columns)
	
	
	# get images
    image_paths = [pp for pp in Path(collection_dir).iterdir() if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
    
    #stores the tuples
    tasks = []


	# loop thru each image path 
    for pp in image_paths:
    
    	#loop thru the possible transformations
        for transform_name in all_transforms_dict.keys():
        	#defines the maximum possible intensity
            max_level = 80
            
            #for each image create a new image where the intensity is increased by 1%. Per 1% increase we create a new image. 
            for level in range(1, max_level + 1): 
                level_value = level / 100
                
                #system stuff to create images etc 
                output_dir = main_output_dir / transform_name
                output_dir.mkdir(parents=True, exist_ok=True)
                #select the row for each image
                row = df[df[img_filename_key] == pp.name].iloc[0].copy()
                
                #add the tuple
                tasks.append((pp, output_dir, transform_name, level_value, row))

    # Use multiprocessing Pool to process images in parallel
    num_workers = cpu_count()
    start_time = time.time()
    
    #create a pool of worker
    with Pool(num_workers) as pool:
        results = []
        
        #process the tasks in parallel 
        for result in tqdm(pool.imap_unordered(augment_and_save_image, tasks), total=len(tasks)):
        	#all valid non null/empty results are stored in the results array
            if result is not None:
                results.append(result)

	#record end time, write to the new csv, and print the runtime.
	
    end_time = time.time()
    augmented_df = pd.DataFrame(results)
    augmented_df.to_csv(augmented_data_csv_path, index=False)

    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.2f} seconds")

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
