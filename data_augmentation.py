"""
Script to generate images with varying intensities of transformations.

This script processes a list of image paths, applies a series of transformations,
and generates new images with incremental intensity levels for each transformation.
The intensity increases by 5% per step, up to a maximum of 80%.

Amount of space needed:
- Each image takes approximately 584 KiB.
- For 784 images, the total storage required for images is approximately:
  - 457,856 KiB
  - 447.33 MiB
  - 0.44 GiB
- The CSV file is initially 12 KiB and adds a new row for each generated image.
- Additional storage for the CSV file with 784 new rows is approximately:
  - 24 KiB
- The storage for directories is approximately:
  - 9 KiB

Total storage needed:
- 457,889 KiB
- 447.35 MiB
- 0.44 GiB

Constant Cost:
- 21 KiB
- 0.02 MiB
- 0.00002 GiB

Dynamic Cost :
- 457,868 KiB
- 447.33 MiB
- 0.44 GiB

Dependencies:
- Ensure the following modules and data structures are available:
  - image_paths: List of image file paths.
  - all_transforms_dict: Dictionary containing transformation names and their corresponding functions.
  - main_output_dir: Main directory where output images will be stored.
  - df: DataFrame containing image metadata, with img_filename_key as the key column.

"""



import argparse
import traceback
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import random
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import albumentations as A
from torchvision import transforms
from collections import deque 



#chatgpt
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


#chatgpt
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

def horizontal_flip(image, level=1.0):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def random_crop(image, level=0.5):
    level = min(max(level, 0.01), 1.0)
    crop_size = int(level * min(image.size))
    return transforms.RandomCrop(crop_size)(image)


#chatgpt
def color_jitter_fn(image, level):
    """
    Applies a color jitter transformation to an image by adjusting brightness, contrast, saturation, and hue.

    Args:
    - image (PIL.Image.Image): The input image to be transformed.
    - level (float): The level or intensity of the color jitter. This should be a value between 0 and 1.

    Returns:
    - PIL.Image.Image: The transformed image with adjusted color properties.
    """
    # Ensure the level is within the expected range
    level = max(0, min(level, 1))

    # Define the factors for brightness, contrast, saturation, and hue based on the level
    brightness_factor = 1 + (level * 0.5)
    contrast_factor = 1 + (level * 0.5)
    saturation_factor = 1 + (level * 0.5)
    hue_factor = level * 0.1

    # Define the transformation using torchvision.transforms
    transform = transforms.ColorJitter(
        brightness=brightness_factor,
        contrast=contrast_factor,
        saturation=saturation_factor,
        hue=hue_factor
    )

    # Apply the transformation to the image
    transformed_image = transform(image)
    return transformed_image



def compose_transformations(transformations):
    """
    Composes a sequence of transformations into a single transformation function.

    Args:
    - transformations (list): A list of transformation functions.

    Returns:
    - function: A function that applies the composed transformations to an image.
    """
    def composed_transformation(image, level):
        for transform in transformations:
            image = transform(image, level)
        return image
    return composed_transformation

# Define both individual and composed transformations
individual_transforms_with_level = {
    "blur": add_blur_fn,
    "color_jitter": color_jitter_fn,
    "random_crop": random_crop,    
    "brightness": adjust_brightness_fn,
    "contrast": adjust_contrast_fn,
    "shadow": add_shadow,
    "time_of_day_dusk": time_of_day_transform_dusk,
    "time_of_day_dawn": time_of_day_transform_dawn,
    "elastic_transform": add_elastic_transform,
    "lens_distortion": add_lens_distortion,
    "noise": add_noise
}

individual_transformations_without_level = {
    "horizontal_flip": horizontal_flip,
}

composed_transforms = {
    "random_crop_elastic_distortion": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["elastic_transform"]
    ]),
    "random_crop_color_jitter": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["color_jitter"]
    ]),
    "random_crop_brightness_adjustment": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["brightness"]
    ]),
    "random_crop_shadow_effect": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["shadow"]
    ]),
    "random_crop_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    "elastic_distortion_color_jitter": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["color_jitter"]
    ]),
    "elastic_distortion_brightness_adjustment": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["brightness"]
    ]),
    "elastic_distortion_shadow_effect": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["shadow"]
    ]),
    "elastic_distortion_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    "elastic_distortion_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    "elastic_distortion_lens_distortion": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    "elastic_distortion_motion_blur": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["blur"]
    ]),
    "elastic_distortion_sensor_noise": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["noise"]
    ]),
    "color_jitter_shadow_effect": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["shadow"]
    ]),
    "color_jitter_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    "color_jitter_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    "color_jitter_lens_distortion": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    "color_jitter_motion_blur": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["blur"]
    ]),
    "color_jitter_sensor_noise": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["noise"]
    ]),
    "brightness_adjustment_shadow_effect": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["shadow"]
    ]),
    "brightness_adjustment_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    "brightness_adjustment_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    "brightness_adjustment_lens_distortion": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    "brightness_adjustment_motion_blur": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["blur"]
    ]),
    "shadow_effect_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    "shadow_effect_lens_distortion": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    "shadow_effect_motion_blur": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["blur"]
    ]),
    "shadow_effect_sensor_noise": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["noise"]
    ]),
    "time_of_day_dusk_lens_distortion": compose_transformations([
        individual_transforms_with_level["time_of_day_dusk"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    "time_of_day_dusk_motion_blur": compose_transformations([
        individual_transforms_with_level["time_of_day_dusk"],
        individual_transforms_with_level["blur"]
    ]),
    "time_of_day_dusk_sensor_noise": compose_transformations([
        individual_transforms_with_level["time_of_day_dusk"],
        individual_transforms_with_level["noise"]
    ]),
    "time_of_day_dawn_lens_distortion": compose_transformations([
        individual_transforms_with_level["time_of_day_dawn"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    "time_of_day_dawn_motion_blur": compose_transformations([
        individual_transforms_with_level["time_of_day_dawn"],
        individual_transforms_with_level["blur"]
    ]),
    "time_of_day_dawn_sensor_noise": compose_transformations([
        individual_transforms_with_level["time_of_day_dawn"],
        individual_transforms_with_level["noise"]
    ]),
    "lens_distortion_motion_blur": compose_transformations([
        individual_transforms_with_level["lens_distortion"],
        individual_transforms_with_level["blur"]
    ]),
    "lens_distortion_sensor_noise": compose_transformations([
        individual_transforms_with_level["lens_distortion"],
        individual_transforms_with_level["noise"]
    ]),
    "motion_blur_sensor_noise": compose_transformations([
        individual_transforms_with_level["blur"],
        individual_transforms_with_level["noise"]
    ])
}

all_transforms_dict = {**individual_transforms_with_level, **individual_transformations_without_level, **composed_transforms}


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
    
    # Unpack the arguments
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

        # Save the new image path and row data to the CSV file
        row['image name'] = save_path.name
        return row
    except Exception as e:
        # Print error message and traceback if an exception occurs
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
    
    # Initialize deque for tasks
    tasks = deque()


	#chatgpt
    # Loop through each image path
    for pp in image_paths:
    
        # Loop through the possible transformations
        for transform_name in all_transforms_dict.keys():
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

    # Use multiprocessing Pool to process images in parallel
    num_workers = cpu_count()
    start_time = time.time()
    
    # Create a pool of workers
    with Pool(num_workers) as pool:
        results = []
        
        # Process the tasks in parallel and collect the results
        for result in tqdm(pool.imap_unordered(augment_and_save_image, tasks), total=len(tasks)):
            # Append valid (non-null) results to the results list
            if result is not None:
                results.append(result)

    # Record end time, write results to the new CSV file, and print the runtime
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
    
    # Determine the collection directories based on the specified level
    if level == 'rosbotxl_data':
        # If processing the entire dataset, get all subdirectories
        collection_dirs = [p for p in Path(parentdir).iterdir() if p.is_dir()]
    elif level == 'collection':
        # If processing a single collection, use the parent directory itself
        collection_dirs = [Path(parentdir)]

    # Process each collection directory
    for collection_dir in collection_dirs:
        process_collection_dir(collection_dir, img_filename_key)

if __name__ == '__main__':

	
	# Parse command line args into 'args' variable
	parser = argparse.ArgumentParser()
	parser.add_argument("--parentdir", type=str, default='/home/husarion/media/usb/rosbotxl_data')
	parser.add_argument("--img_filename_key", type=str, default="image name")
	parser.add_argument("--level", type=str, choices=['rosbotxl_data', 'collection'], default='rosbotxl_data',
                    	help="Specify the directory level to process: 'rosbotxl_data' for the whole dataset or 'collection' for a single collection.")
                    	
	# Transformations that require an intensity level
	parser.add_argument("-b", "--blur", type=float, default=0.0, help="Intensity level for blur effect, between 0.0 (no blur) and 1.0 (maximum blur).")
	parser.add_argument("-j", "--color_jitter", type=float, default=0.0, help="Intensity level for color jitter effect, between 0.0 (no effect) and 1.0 (maximum effect).")
	parser.add_argument("-c", "--random_crop", type=float, default=0.0, help="Intensity level for random crop, between 0.0 (no crop) and 1.0 (maximum crop size).")
	parser.add_argument("-n", "--brightness", type=float, default=1.0, help="Intensity level for brightness adjustment, between 0.0 (dark) and 2.0 (bright).")
	parser.add_argument("-t", "--contrast", type=float, default=1.0, help="Intensity level for contrast adjustment, between 0.0 (low contrast) and 2.0 (high contrast).")
	parser.add_argument("-s", "--shadow", type=float, default=0.0, help="Intensity level for adding shadows, between 0.0 (no shadow) and 1.0 (maximum shadow).")
	parser.add_argument("-d", "--time_of_day_dusk", type=float, default=0.0, help="Intensity level for dusk time of day effect, between 0.0 (day) and 1.0 (dusk).")
	parser.add_argument("-a", "--time_of_day_dawn", type=float, default=0.0, help="Intensity level for dawn time of day effect, between 0.0 (night) and 1.0 (dawn).")
	parser.add_argument("-e", "--elastic_transform", type=float, default=0.0, help="Intensity level for elastic transformation, between 0.0 (no distortion) and 1.0 (maximum distortion).")
	parser.add_argument("-l", "--lens_distortion", type=float, default=0.0, help="Intensity level for lens distortion effect, between 0.0 (no distortion) and 1.0 (noticeable distortion).")
	parser.add_argument("-o", "--noise", type=float, default=0.0, help="Intensity level for adding noise, between 0.0 (clean) and 1.0 (noisy).")
	
	# Transformations that do not require an intensity level
	parser.add_argument("-f", "--horizontal_flip", action='store_true', help="Apply horizontal flip to the images.")
	
	
	args = parser.parse_args()
	print("args:" + str(args))

	
    process_parent_dir(args.parentdir, args.level, args.img_filename_key)
