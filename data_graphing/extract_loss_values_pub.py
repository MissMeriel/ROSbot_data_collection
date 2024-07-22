# Extract the loss values from a slurm out file and put them into an array
# makes a txt file with the array in it
# Intended to use with graph_loss_pub.m

import re
import numpy as np

# Function to extract loss values from a .out file
def extract_loss_values(file_path):
    loss_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'loss:\s*([\d.]+)', line)
            if match:
                loss_values.append(float(match.group(1)))
    return loss_values

# Function to create the mean_loss_array
def create_mean_loss_array(loss_values):
    indices = np.arange(1, len(loss_values) + 1).reshape(-1, 1)
    mean_loss_array = np.hstack((indices, np.array(loss_values).reshape(-1, 1)))
    return mean_loss_array

# Example usage
file_path = 'slurm-4875828.out'  # Replace with your .out file path
output_path = 'slurm-4875828.txt'

loss_values = extract_loss_values(file_path)
mean_loss_array = create_mean_loss_array(loss_values)

# Save mean_loss_array to a text file
np.savetxt(output_path, mean_loss_array, fmt='%.8f')

print(mean_loss_array)