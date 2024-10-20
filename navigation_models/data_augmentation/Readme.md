
# Image Augmentation and Processing

This project is designed to perform image augmentation and processing on a dataset of images. It applies various transformations and composed transformations to the images and updates a CSV file with the new image paths.

- `main.py`: The main script that parses command-line arguments and initiates the image processing.
- `processing.py`: Contains functions for processing collections and parent directories of images, applying transformations, and saving the augmented images.
- `transformations.py`: Contains the definitions of individual and composed transformations that can be applied to images.
- `README.md`: This file.


## Usage

### Command-Line Arguments

- `--parentdir`: Specifies the parent directory containing the images to process.
- `--img_filename_key`: Specifies the key used in the CSV file to identify the image filenames.
- `--level`: Specifies the level of processing (either `rosbotxl_data` for the whole dataset or `collection` for a single collection).
- `--transformations`: List of transformations to apply. Example: `--transformations blur contrast horizontal_flip`
- `--composed_transforms`: List of composed transformations to apply. Example: `--composed_transforms blur,contrast random_crop,brightness`
- `--specify`: (DONT USE THIS ITS VERY BUGGY NEEDS A QUICK FIX) List of specific image paths to process.

### Example Commands

#### Basic Usage

Process the entire dataset applying blur and contrast transformations:
```bash
python main.py --parentdir /path/to/parentdir --img_filename_key "image name" --level rosbotxl_data --transformations blur contrast
```

#### Using Composed Transformations

Apply composed transformations like blur and contrast, as well as random crop and brightness:
```bash
python main.py --parentdir /path/to/parentdir --img_filename_key "image name" --level rosbotxl_data --composed_transforms blur,contrast random_crop,brightness
```

#### Processing Specific Images

Process only specified images:
```bash
python main.py --parentdir /path/to/parentdir --img_filename_key "image name" --level rosbotxl_data --specify /path/to/image1.jpg /path/to/image2.jpg
```

#### Full Example

Apply multiple transformations and composed transformations to specified images:
```bash
python main.py --parentdir /path/to/parentdir --img_filename_key "image name" --level rosbotxl_data --transformations blur contrast horizontal_flip --composed_transforms blur,contrast random_crop,brightness --specify /path/to/image1.jpg /path/to/image2.jpg
```

## File Descriptions

### main.py
The entry point of the application. It parses command-line arguments and initiates the image processing by calling the `process_parent_dir` function from `processing.py`.

### processing.py
Contains the core functionality for processing images:
- `augment_and_save_image(args)`: Applies a specified transformation to an image and saves the result.
- `augment_and_save_composed_image(args)`: Applies a composed transformation to an image and saves the result.
- `process_collection_dir(collection_dir, img_filename_key, transformations_list, composed_transformations_list, specified_images)`: Processes a collection directory of images, applying the specified transformations.
- `process_parent_dir(parentdir, level, img_filename_key, transformations_list, composed_transformations_list, specified_images)`: Processes the parent directory, calling `process_collection_dir` for each collection.

### transformations.py
Defines the individual and composed transformations that can be applied to images:
- Individual transformations such as `add_shadow`, `time_of_day_transform_dusk`, `add_noise`, etc.
- Function `compose_transformations(transformations)`: Composes multiple transformations into a single function.

## Additional Notes

- Ensure that the paths specified in the command-line arguments exist and are accessible.
- The transformations will be saved in an `augmented_data` directory within each collection directory.
- The CSV file corresponding to each collection will be updated with the new image paths.
