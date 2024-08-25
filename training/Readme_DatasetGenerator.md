
# DatasetGenerator.py

## Requirements

To run this project, you need the following dependencies:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Torch
- Torchvision
- Kornia
- Scipy
- PIL
- Skimage
- OpenCV

You can install the required packages using pip:

```sh
pip install numpy pandas matplotlib torch torchvision kornia scipy pillow scikit-image opencv-python
```

## Files and Directories

- `data_augmentation/transformations.py`: Contains the data augmentation functions.
- `train.py`: Main script for training the model.
- `DataSequence.py`: Contains the `DataSequence` and `MultiDirectoryDataSequence` classes for loading and augmenting the dataset.

## Usage

### Data Augmentation

Data augmentation is a critical aspect of this project. Various augmentation techniques are applied to the images to make the model more robust to different driving conditions. The following transformations are used:

- **Shadows**: Adds shadows to simulate different lighting conditions.
- **Time of Day**: Simulates different times of day, such as dusk.
- **Elastic Transform**: Applies elastic transformations to the images.
- **Blur**: Adds blur to simulate motion blur or foggy conditions.
- **Color Jitter**: Randomly changes the brightness, contrast, saturation, and hue of the images.
- **Brightness Adjustment**: Adjusts the brightness of the images.
- **Contrast Adjustment**: Adjusts the contrast of the images.
- **Saturation Adjustment**: Adjusts the saturation of the images.
- **Horizontal Flip**: Flips the images horizontally to simulate different driving directions.
- **Lens Distortion**: Adds lens distortion to the images.
- **Noise**: Adds random noise to the images.

#### How Data Augmentation Works

1. **Loading Images**: Images are loaded using the `PIL` library and converted to arrays using `numpy` or `torch`.
2. **Transformation Functions**: Individual and composed transformation functions are defined. Each function takes an image as input and applies a specific augmentation.
3. **Applying Transformations**: The transformations are applied to the images. Both individual and composed transformations are used to ensure a wide variety of augmented images.
4. **Handling Augmented Images**: The augmented images are converted back to tensors and stored in a list. These tensors are then used in the training process.

### Example of Augmentation Process

Here’s a brief overview of how the augmentation process is implemented:

1. **Define Transformation Functions**: Functions like `add_shadow`, `time_of_day_transform_dusk`, `add_blur_fn`, etc., are defined in the `data_augmentation/transformations.py` file.
2. **Custom Transform Function**: A helper function `custom_transform` applies individual transformations to the images.
3. **Composed Transform Function**: Another helper function `apply_composed_transformations` applies a sequence of transformations to the images.
4. **Augmenting Data**: In the `MultiDirectoryDataSequence` class, these helper functions are used to augment the images. The augmented images are then included in the dataset.

### Training the Model

1. **Prepare your dataset**: Ensure your dataset is organized in directories where each directory contains images and a file with corresponding steering angles and other sensor data.

2. **Run the training script**:
    ```sh
    python train.py <dataset_directory> --batch <batch_size> --epochs <num_epochs> --lr <learning_rate> --robustification <True/False> --noisevar <noise_variance> --log_interval <log_interval>
    ```

   Example:
    ```sh
    python train.py ./data --batch 16 --epochs 100 --lr 1e-4 --robustification True --noisevar 15 --log_interval 50
    ```

### Arguments

- `dataset`: Parent directory of the training dataset.
- `--batch`: Batch size for training (default: 16).
- `--epochs`: Number of training epochs (default: 100).
- `--lr`: Learning rate for the optimizer (default: 1e-4).
- `--robustification`: Whether to apply robustification techniques (default: True).
- `--noisevar`: Noise variance for robustification (default: 15).
- `--log_interval`: Interval for logging training progress (default: 50).

### Training Process

The training process involves multiple epochs. An epoch refers to one complete pass through the entire training dataset. During each epoch, the model learns and updates its parameters based on the training data.

#### Steps in Each Epoch:

1. **Data Loading**: The training dataset is loaded in batches as specified by the batch size.
2. **Forward Pass**: The input images are passed through the DNN to obtain the predicted steering angles.
3. **Loss Calculation**: The Mean Squared Error (MSE) loss between the predicted and actual steering angles is computed.
4. **Backward Pass**: The gradients of the loss with respect to the model parameters are calculated using backpropagation.
5. **Parameter Update**: The optimizer updates the model parameters to minimize the loss.
6. **Logging**: The training progress, including the running loss, is logged at specified intervals.

### Training Output

During training, the script logs the following information:
- Moments of the output distribution.
- Total number of samples.
- Running loss at specified intervals.
- Best model saved based on lowest Mean Squared Error (MSE) loss.
- Final trained model saved to a specified directory.
- Meta information about the training process saved to a text file.

## Classes and Functions

### `DataSequence(data.Dataset)`

A dataset class that loads images and corresponding steering angles from a directory.

### `MultiDirectoryDataSequence(data.Dataset)`

A dataset class that loads images and corresponding steering angles from multiple directories. It applies data augmentation techniques to the images.

### Data Augmentation Functions

The following functions are used for data augmentation:

- `add_shadow()`
- `time_of_day_transform_dusk()`
- `add_elastic_transform()`
- `add_blur_fn()`
- `color_jitter_fn()`
- `adjust_brightness_fn()`
- `adjust_contrast_fn()`
- `adjust_saturation_fn()`
- `horizontal_flip()`
- `add_lens_distortion()`
- `add_noise()`

## Notes

- Ensure the dataset is properly formatted and preprocessed before starting the training.
- Adjust the hyperparameters (batch size, learning rate, etc.) as needed based on your specific requirements and dataset characteristics.

## License

This project is licensed under the MIT License.

---

Happy training! If you encounter any issues or have any questions, feel free to open an issue or reach out for support.
