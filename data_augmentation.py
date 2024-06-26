
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

# Define the argument parser
parser = argparse.ArgumentParser(description="Script to generate images with varying intensities of transformations.")

# Add arguments to the parser
parser.add_argument("--parentdir", "-p", type=str, required=True, help="Main directory where output images will be stored.")
parser.add_argument("--img_filename_key", "-k", type=str, default="image name", help="Key in the CSV file that corresponds to image filenames.")
parser.add_argument("--level", "-l", type=str, choices=['rosbotxl_data', 'collection'], default='rosbotxl_data', help="Specify the directory level to process: 'rosbotxl_data' for the whole dataset or 'collection' for a single collection.")
parser.add_argument("--intensity", "-i", type=float, default=0.5, help="Intensity level of the transformation.")
parser.add_argument("--noise_level", "-n", type=float, default=0.05, help="Standard deviation of the Gaussian noise to be added to LiDAR data.")
parser.add_argument("--max_level", "-m", type=int, default=80, help="Maximum intensity level for transformations.")
parser.add_argument("--step", "-s", type=int, default=5, help="Step size for increasing intensity levels.")
parser.add_argument("--num_workers", "-w", type=int, default=cpu_count(), help="Number of parallel workers for processing images.")

# Add transformation flags
transform_flags = parser.add_argument_group('transformations')
transform_flags.add_argument("-b", "--blur", action="store_true", help="Apply blur transformation.")
transform_flags.add_argument("-c", "--color_jitter", action="store_true", help="Apply color jitter transformation.")
transform_flags.add_argument("-r", "--random_crop", action="store_true", help="Apply random crop transformation.")
transform_flags.add_argument("-B", "--brightness", action="store_true", help="Apply brightness adjustment.")
transform_flags.add_argument("-C", "--contrast", action="store_true", help="Apply contrast adjustment.")
transform_flags.add_argument("-S", "--shadow", action="store_true", help="Apply shadow effect.")
transform_flags.add_argument("-d", "--time_of_day_dusk", action="store_true", help="Apply dusk lighting condition.")
transform_flags.add_argument("-D", "--time_of_day_dawn", action="store_true", help="Apply dawn lighting condition.")
transform_flags.add_argument("-e", "--elastic_transform", action="store_true", help="Apply elastic transformation.")
transform_flags.add_argument("-L", "--lens_distortion", action="store_true", help="Apply lens distortion.")
transform_flags.add_argument("-N", "--noise", action="store_true", help="Apply noise transformation.")
transform_flags.add_argument("-H", "--horizontal_flip", action="store_true", help="Apply horizontal flip.")

# Parse the arguments
args = parser.parse_args()


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
    
    
    
 	#chatgpt - make a function that adds gaussian noise to my lidar data. Here is how my lidar data is stored and here is how I flipped the lidar data.
def add_lidar_noise(lidar_ranges_str, noise_level=0.05):
    """
    Adds Gaussian noise to the LiDAR readings.

    Args:
    - lidar_ranges_str (str): Original LiDAR readings as a space-separated string.
    - noise_level (float): Standard deviation of the Gaussian noise to be added. Default is 0.05.

    Returns:
    - str: LiDAR readings with added noise, as a space-separated string.
    """
    # Split the 'lidar_ranges' string into a list of string values
    lidar_ranges_list = lidar_ranges_str.split()
    
    # Convert each string value in the list to a float
    lidar_ranges = []
    for range_value in lidar_ranges_list:
        lidar_ranges.append(float(range_value))
    
    # Convert the list to a NumPy array for easier manipulation
    lidar_ranges_array = np.array(lidar_ranges)
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, size=lidar_ranges_array.shape)
    
    # Add the noise to the LiDAR readings
    noisy_lidar_ranges = lidar_ranges_array + noise
    
    # Ensure the LiDAR readings are within a realistic range (e.g., non-negative)
    noisy_lidar_ranges = np.clip(noisy_lidar_ranges, a_min=0, a_max=None)
    
    # Convert the NumPy array back to a list of strings
    noisy_lidar_ranges_list = [str(value) for value in noisy_lidar_ranges]
    
    # Join the list into a space-separated string
    noisy_lidar_ranges_str = ' '.join(noisy_lidar_ranges_list)
    
    return noisy_lidar_ranges_str

# Define both individual and composed transformations
# Compose transformations to apply multiple effects sequentially
composed_transforms = {


    "random_crop_elastic_distortion": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["elastic_transform"]
    ]),
    # Randomly crops the image to simulate different viewing angles or distances,
    # then applies elastic distortion to simulate realistic deformations.
    
    "random_crop_brightness_adjustment": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["brightness"]
    ]),
    # Randomly crops the image to change the perspective, followed by adjusting brightness 
    # to simulate varying lighting conditions.

    "random_crop_shadow_effect": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["shadow"]
    ]),
    # Randomly crops the image to simulate different perspectives and then adds shadows 
    # to simulate real-world lighting effects.

    "random_crop_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["random_crop"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    # Randomly crops the image to change the perspective, followed by adjusting the colors 
    # to simulate dusk lighting conditions.

    "elastic_distortion_color_jitter": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["color_jitter"]
    ]),
    # Applies elastic distortion to simulate realistic deformations and then adjusts color 
    # properties to simulate different lighting conditions.

    "elastic_distortion_shadow_effect": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["shadow"]
    ]),
    # Applies elastic distortion to simulate realistic deformations, then adds shadows to 
    # simulate real-world lighting effects.

    "elastic_distortion_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    # Applies elastic distortion to simulate realistic deformations, followed by adjusting 
    # the colors to simulate dusk lighting conditions.

    "elastic_distortion_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    # Applies elastic distortion to simulate realistic deformations, followed by adjusting 
    # the colors to simulate dawn lighting conditions.


    "elastic_distortion_motion_blur": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["blur"]
    ]),
    # Applies elastic distortion to simulate realistic deformations and then adds motion 
    # blur to simulate camera movement.

    "elastic_distortion_sensor_noise": compose_transformations([
        individual_transforms_with_level["elastic_transform"],
        individual_transforms_with_level["noise"]
    ]),
    # Applies elastic distortion to simulate realistic deformations and then adds sensor 
    # noise to simulate real-world sensor imperfections.

    "color_jitter_shadow_effect": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["shadow"]
    ]),
    # Adjusts color properties to simulate different lighting conditions and then adds 
    # shadows to enhance real-world lighting effects.

    "color_jitter_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    # Adjusts color properties to simulate different lighting conditions, followed by 
    # adjusting colors to simulate dusk lighting conditions.

    "color_jitter_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    # Adjusts color properties to simulate different lighting conditions, followed by 
    # adjusting colors to simulate dawn lighting conditions.


    "color_jitter_motion_blur": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["blur"]
    ]),
    # Adjusts color properties to simulate different lighting conditions, followed by 
    # adding motion blur to simulate camera movement.

    "color_jitter_sensor_noise": compose_transformations([
        individual_transforms_with_level["color_jitter"],
        individual_transforms_with_level["noise"]
    ]),
    # Adjusts color properties to simulate different lighting conditions, followed by 
    # adding sensor noise to simulate real-world sensor imperfections.

    "brightness_adjustment_shadow_effect": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["shadow"]
    ]),
    # Adjusts brightness to handle varying lighting conditions, followed by adding shadows 
    # to simulate real-world lighting effects.

    "brightness_adjustment_time_of_day_dusk": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["time_of_day_dusk"]
    ]),
    # Adjusts brightness to handle varying lighting conditions, followed by adjusting colors 
    # to simulate dusk lighting conditions.

    "brightness_adjustment_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    # Adjusts brightness to handle varying lighting conditions, followed by adjusting colors 
    # to simulate dawn lighting conditions.

    "brightness_adjustment_motion_blur": compose_transformations([
        individual_transforms_with_level["brightness"],
        individual_transforms_with_level["blur"]
    ]),
    # Adjusts brightness to handle varying lighting conditions, followed by adding motion 
    # blur to simulate camera movement.

    "shadow_effect_time_of_day_dawn": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["time_of_day_dawn"]
    ]),
    # Adds shadows to simulate real-world lighting effects, followed by adjusting colors to 
    # simulate dawn lighting conditions.

    "shadow_effect_motion_blur": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["blur"]
    ]),
    # Adds shadows to simulate real-world lighting effects, followed by adding motion blur 
    # to simulate camera movement.

    "shadow_effect_sensor_noise": compose_transformations([
        individual_transforms_with_level["shadow"],
        individual_transforms_with_level["noise"]
    ]),
    # Adds shadows to simulate real-world lighting effects, followed by adding sensor noise 
    # to simulate real-world sensor imperfections.

    "time_of_day_dusk_lens_distortion": compose_transformations([
        individual_transforms_with_level["time_of_day_dusk"],
        individual_transforms_with_level["lens_distortion"]
    ]),
    # Adjusts colors to simulate dusk lighting conditions, followed by adding lens distortion 
    # to handle different camera lenses.

    "time_of_day_dusk_motion_blur": compose_transformations([
        individual_transforms_with_level["time_of_day_dusk"],
        individual_transforms_with_level["blur"]
    ]),
    # Adjusts colors to simulate dusk lighting conditions, followed by adding motion blur to 
    # simulate camera movement.

    "time_of_day_dusk_sensor_noise": compose_transformations([
        individual_transforms_with_level["time_of_day_dusk"],
        individual_transforms_with_level["noise"]
    ]),
    # Adjusts colors to simulate dusk lighting conditions, followed by adding sensor noise 
    # to simulate real-world sensor imperfections.


    "time_of_day_dawn_motion_blur": compose_transformations([
        individual_transforms_with_level["time_of_day_dawn"],
        individual_transforms_with_level["blur"]
    ]),
    # Adjusts colors to simulate dawn lighting conditions, followed by adding motion blur to 
    # simulate camera movement.

    "time_of_day_dawn_sensor_noise": compose_transformations([
        individual_transforms_with_level["time_of_day_dawn"],
        individual_transforms_with_level["noise"]
    ]),
    # Adjusts colors to simulate dawn lighting conditions, followed by adding sensor noise 
    # to simulate real-world sensor imperfections.

    "motion_blur_sensor_noise": compose_transformations([
        individual_transforms_with_level["blur"],
        individual_transforms_with_level["noise"]
    ]),
    # Adds motion blur to simulate camera movement, followed by adding sensor noise to 
    # simulate real-world sensor imperfections.

    "horizontal_flip_motion_blur_elastic": compose_transformations([
        individual_transformations_without_level["horizontal_flip"],
        individual_transforms_with_level["blur"],
        individual_transforms_with_level["elastic_transform"]
    ])
    # Flips the image horizontally to augment the dataset, followed by adding motion blur to 
    # simulate camera movement and applying elastic distortion to simulate realistic deformations.
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
            if pd.notnull(row['lidar_ranges']):
                # Split the 'lidar_ranges' string into a list of string values
                lidar_ranges_str = row['lidar_ranges'].split()
                
                # Initialize an empty list to hold the float values
                lidar_ranges = []
                
                # Convert each string value in the list to a float and append to the lidar_ranges list
                for range_value in lidar_ranges_str:
                    lidar_ranges.append(float(range_value))
                
                # Reverse the list to match the horizontal flip transformation
                lidar_ranges.reverse()
                
                # Convert the list back to a space-separated string
                row['lidar_ranges'] = ' '.join(map(str, lidar_ranges))

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
    
    # List to store tasks for multiprocessing
    tasks = []

    # Check if any transformation flags are set
    transformations = []
    if args.blur: transformations.append("blur")
    if args.color_jitter: transformations.append("color_jitter")
    if args.random_crop: transformations.append("random_crop")
    if args.brightness: transformations.append("brightness")
    if args.contrast: transformations.append("contrast")
    if args.shadow: transformations.append("shadow")
    if args.time_of_day_dusk: transformations.append("time_of_day_dusk")
    if args.time_of_day_dawn: transformations.append("time_of_day_dawn")
    if args.elastic_transform: transformations.append("elastic_transform")
    if args.lens_distortion: transformations.append("lens_distortion")
    if args.noise: transformations.append("noise")
    if args.horizontal_flip: transformations.append("horizontal_flip")

    # If no specific transformations are provided, use all transformations
    if not transformations:
        transformations = list(all_transforms_dict.keys())

    for pp in image_paths:
        for level in range(args.step, args.max_level + args.step, args.step):
            level_value = level / 100
            output_dir = main_output_dir / '_'.join(transformations)
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
            row = df[df[img_filename_key] == pp.name].iloc[0].copy()  # Get the corresponding row from the DataFrame
            tasks.append((pp, output_dir, transformations, level_value, row))

    num_workers = args.num_workers
    start_time = time.time()

    # Use multiprocessing to process the tasks in parallel
    with Pool(num_workers) as pool:
        results = [result for result in tqdm(pool.imap_unordered(augment_and_save_image, tasks), total=len(tasks)) if result is not None]

    end_time = time.time()
    augmented_df = pd.DataFrame(results)  # Create a DataFrame from the results
    augmented_df.to_csv(augmented_data_csv_path, index=False)  # Save the augmented data to a CSV file

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
    process_parent_dir(args.parentdir, args.level, args.img_filename_key)

