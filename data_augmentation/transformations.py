# transformations.py

from PIL import Image, ImageEnhance
import numpy as np
import random
import albumentations as A
from torchvision import transforms

# Define image transformation functions
def add_shadow(image, level=0.5):
    # Apply shadow to the image
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
    # Simulate dusk by reducing color saturation
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 - level)

def time_of_day_transform_dawn(image, level=0.5):
    # Simulate dawn by increasing color saturation
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1 + level)

def add_elastic_transform(image, level=0.5):
    # Apply elastic deformation to the image
    level = min(max(level, 0.01), 1.0)
    image = np.array(image)
    transform = A.ElasticTransform(alpha=level * 100, sigma=level * 10, alpha_affine=level * 10, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_lens_distortion(image, level=0.5):
    # Apply lens distortion to the image
    level = min(max(level, 0.01), 1.0)
    image = np.array(image)
    transform = A.OpticalDistortion(distort_limit=0.2 * level, shift_limit=0.2 * level, p=1.0)
    augmented = transform(image=image)
    return Image.fromarray(augmented['image'])

def add_noise(image, level=0.5):
    # Add random noise to the image
    level = min(max(level, 0.01), 1.0)
    array = np.array(image)
    noise = np.random.normal(0, level * 255, array.shape)
    array = np.clip(array + noise, 0, 255)
    return Image.fromarray(array.astype('uint8'))

def add_blur_fn(img, level=0.5):
    # Apply Gaussian blur to the image
    level = min(max(level, 0.01), 1.0)
    kernel_size = int(np.ceil(level * 8))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 5))(img)

def adjust_brightness_fn(img, level=0.5):
    # Adjust the brightness of the image
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(level)

def adjust_contrast_fn(img, level=0.5):
    # Adjust the contrast of the image
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(level)

def adjust_saturation_fn(img, level=0.5):
    # Adjust the saturation of the image
    level = min(max(level, 0.01), 1.0)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(level)

def horizontal_flip(image, level=1.0):
    # Flip the image horizontally
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def random_crop(image, level=0.5):
    # Randomly crop the image
    level = min(max(level, 0.01), 1.0)
    crop_size = int(level * min(image.size))
    return transforms.RandomCrop(crop_size)(image)

def color_jitter_fn(image, level):
    """
    Applies a color jitter transformation to an image by adjusting brightness, contrast, saturation, and hue.

    Args:
    - image (PIL.Image.Image): The input image to be transformed.
    - level (float): The level or intensity of the color jitter. This should be a value between 0 and 1.

    Returns:
    - PIL.Image.Image: The transformed image with adjusted color properties.
    """
    level = max(0, min(level, 1))
    brightness_factor = 1 + (level * 0.5)
    contrast_factor = 1 + (level * 0.5)
    saturation_factor = 1 + (level * 0.5)
    hue_factor = level * 0.1
    transform = transforms.ColorJitter(
        brightness=brightness_factor,
        contrast=contrast_factor,
        saturation=saturation_factor,
        hue=hue_factor
    )
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

all_transforms_dict = {**individual_transforms_with_level, **individual_transformations_without_level}
