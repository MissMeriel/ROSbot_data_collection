
# train_DAVE2.py


## Requirements

To run this project, you need the following dependencies:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Torch
- Torchvision
- argparse

You can install the required packages using pip:

\`\`\`sh
pip install numpy pandas matplotlib torch torchvision argparse
\`\`\`

## Files and Directories

- \`Zach_DatasetGenerator.py\`: Contains the \`MultiDirectoryDataSequence\` class for loading and augmenting the dataset.
- \`Zach_DAVE2pytorch.py\`: Contains the DAVE2 model definitions (\`DAVE2PytorchModel\`, \`DAVE2v1\`, \`DAVE2v2\`, \`DAVE2v3\`, \`Epoch\`).
- \`train.py\`: Main script for training the model.

## Usage

### Training the Model

1. **Prepare your dataset**: Ensure your dataset is organized in directories where each directory contains images and a file with corresponding steering angles.

2. **Run the training script**:
    \`\`\`sh
    python train.py <dataset_directory> --batch <batch_size> --epochs <num_epochs> --lr <learning_rate> --robustification <True/False> --noisevar <noise_variance> --log_interval <log_interval>
    \`\`\`

    Example:
    \`\`\`sh
    python train.py ./data --batch 16 --epochs 100 --lr 1e-4 --robustification True --noisevar 15 --log_interval 50
    \`\`\`

### Arguments

- \`dataset\`: Parent directory of the training dataset.
- \`--batch\`: Batch size for training (default: 16).
- \`--epochs\`: Number of training epochs (default: 100).
- \`--lr\`: Learning rate for the optimizer (default: 1e-4).
- \`--robustification\`: Whether to apply robustification techniques (default: True).
- \`--noisevar\`: Noise variance for robustification (default: 15).
- \`--log_interval\`: Interval for logging training progress (default: 50).

### Training Output

During training, the script logs the following information:
- Moments of the output distribution.
- Total number of samples.
- Running loss at specified intervals.
- Best model saved based on lowest Mean Squared Error (MSE) loss.
- Final trained model saved to a specified directory.
- Meta information about the training process saved to a text file.

## Functions

### \`parse_arguments()\`

Parses command-line arguments for the training script.

### \`characterize_steering_distribution(y_steering, generator)\`

Characterizes the steering distribution by categorizing into turning and straight segments and printing their distribution moments.

### \`main()\`

Main function that orchestrates the training process:
- Initializes the model and dataset.
- Configures the DataLoader.
- Performs the training loop.
- Logs training progress and saves the best model.
- Cleans up models from previous epochs.

## Notes

- Ensure the dataset is properly formatted and preprocessed before starting the training.
- Adjust the hyperparameters (batch size, learning rate, etc.) as needed based on your specific requirements and dataset characteristics.

## License

This project is licensed under the MIT License.

---

Happy training! If you encounter any issues or have any questions, feel free to open an issue or reach out for support.
