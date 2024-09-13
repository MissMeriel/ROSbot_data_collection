import sys
import os
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
import pandas as pd
from PIL import Image
from torch.autograd import Variable
import csv
import re

sys.path.append("../training")
from DatasetGenerator import MultiDirectoryDataSequence
sys.path.append("../models")
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v1, DAVE2v2, DAVE2v3, Epoch


# Parse arguments
def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Generate dataset and plot predictions vs actual")
    parser.add_argument('--dataset_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--models_dir', type=str, help='Directory containing .pt model files')
    parser.add_argument('--output_dir', type=str, default='./inference', help='Directory to save the plots')
    parser.add_argument('--image_width', type=str, default="2560", help='Image width for the dataset')
    parser.add_argument('--image_height', type=str, default="720", help='Image height for the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    args = parser.parse_args()
    return args


# Preload image names and tensors with directory paths
def preload_image_names(dataset_dir):
    image_dict = {}
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file == 'data_cleaned.csv':
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    image_name = row['image name']
                    image_path = os.path.join(root, image_name)
                    image_dict[image_name] = root
    return image_dict


# Generate the dataset
def generate_dataset(dataset_dir, image_size, batch_size):
    dataset = MultiDirectoryDataSequence(
        dataset_dir,
        image_size=image_size,
        transform=Compose([ToTensor()]),
        robustification=False
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


# Calculate memory size and sample size
def calculate_memory_and_sample_size(image_dict):
    total_memory_size = 0
    total_sample_size = len(image_dict)
    print(f"total_sample_size= {total_sample_size}", flush=True)
    for path in image_dict.keys():
        # print(f"path= {path}", flush=True)
        image_dir_path = os.path.join(str(image_dict[path]), str(path))
        # print(f"image_dir_path= {image_dir_path}", flush=True)
        image = Image.open(image_dir_path)
        image_memory = (image.size[0] * image.size[1] * len(image.getbands())) # width x height x channel
        total_memory_size += image_memory
    #     print(f"image_memory= {image_memory} for {image_dir_path}", flush=True)
    print(f"total_memory_size= {total_memory_size}", flush=True)
    return total_sample_size, total_memory_size


# Plot the predictions vs actual values and save to CSV
def plot_predictions(models_dir, data_loader, output_dir, image_size, image_dict):
    # global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_inference_times = []
    model_losses = []

    for model_file in sorted(os.listdir(models_dir), key=lambda x: int(re.findall(r'epoch(\d+)', x)[0]) if re.findall(r'epoch(\d+)', x) else float('inf')):
        if model_file.endswith('.pt'):
            model_path = os.path.join(models_dir, model_file)
            try:
                # model = DAVE2v3(input_shape=(1280, 360))
                model = DAVE2v3(input_shape=image_size).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
            except TypeError:
                try:
                    model = torch.load(model_path, map_location=device)
                except TypeError as e:
                    print(e)

            model.eval()
            print(f"Finished setting the model in evaluation mode for {model_file}", flush=True)
            actual_values = []
            predicted_values = []
            image_file_names = []

            start_model_time = time.time()

            with torch.no_grad():
                def get_image_number(image_path):
                    match = re.findall(r'\d+', str(image_path))
                    return int(match[0]) if match else float('inf')

                flat_data_loader = []
                for hashmap in data_loader:
                    for i in range(len(hashmap['image'])):
                        flat_data_loader.append({
                            'image': hashmap['image'][i],
                            'image name': hashmap['image name'][i],
                            'angular_speed_z': hashmap['angular_speed_z'][i]
                        })

                sorted_image_dict = sorted(flat_data_loader, key=lambda x: get_image_number(x['image']))
                # print(f"flat_data_loader: {flat_data_loader}", flush=True)
                # print(f"sorted_image_dict: {sorted_image_dict}", flush=True)

                for hashmap in sorted_image_dict:
                    images = hashmap['image name'].float().to(device)
                    images = torch.nn.functional.interpolate(images.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
                    image_name = hashmap['image']
                    actual_angular_speed_z = hashmap['angular_speed_z'].float().to(device)

                    outputs = model(images.unsqueeze(0))
                    predicted_angular_speed_z = outputs.squeeze(0)

                    actual_values.extend(actual_angular_speed_z.cpu().numpy())
                    predicted_values.extend(predicted_angular_speed_z.cpu().numpy())

                    # Debug statement to check if image_name is in image_dict
                    if str(image_name) in image_dict:
                        image_file_names.append((str(image_name), image_dict[str(image_name)]))
                    else:
                        print(f"Warning: {str(image_name)} not found in image_dict", flush=True)

            end_model_time = time.time()
            model_time = end_model_time - start_model_time
            print(f"Time: {model_time} is the time it takes for {model_file}", flush=True)
            model_inference_times.append((model_file, model_time))

            mse_loss = F.mse_loss(torch.tensor(predicted_values), torch.tensor(actual_values)).item()
            print(f"Loss: {mse_loss} is the MSE loss for {model_file}", flush=True)
            model_losses.append((model_file, mse_loss))

            # Clear memory
            del images, actual_angular_speed_z, outputs
            torch.cuda.empty_cache()

            # Convert image file names to simple strings for plotting
            plot_image_names = [str(img_name) for img_name, _ in image_file_names]

            # Plot results and save as png
            plt.figure(figsize=(15, 5))
            plt.plot(plot_image_names, actual_values, label='Actual')
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.tick_params(bottom=False)
            plt.plot(plot_image_names, predicted_values, label='Predicted')
            plt.xlabel('Image Index')
            plt.ylabel('Angular Speed Z')
            plt.title(f'Actual vs Predicted Angular Speed Z for {model_file}')
            plt.legend()
            plt.xticks(rotation=90, fontsize=8)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{model_file}_plot.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Finished plotting for {model_file}", flush=True)

            # Save results to CSV
            csv_file_path = os.path.join(output_dir, f'{model_file}_results.csv')
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Image Name', 'Directory Path', 'Actual Angular Speed', 'Predicted Angular Speed'])
                for (img_name, dir_path), actual_speed, predicted_speed in zip(image_file_names, actual_values, predicted_values):
                    writer.writerow([img_name, dir_path, actual_speed, predicted_speed])
            print(f"Finished saving CSV for {model_file}", flush=True)

    # Record the time each model takes in csv file
    with open(os.path.join(output_dir, 'model_inference_time.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model File', 'Inference Time'])
        writer.writerows(model_inference_times)
    print(f"Saved model_inference_time.csv in {output_dir}", flush=True)

    # Record the loss for each model in csv file
    with open(os.path.join(output_dir, 'model_loss.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model File', 'MSE Loss'])
        writer.writerows(model_losses)
    print(f"Saved model_loss.csv in {output_dir}", flush=True)


def get_metainfo(start_time, output_dir, sample_size, total_memory_size):
    time_total = time.time() - start_time
    print(f"Total time for inference: {time_total}", flush=True)
    # save the meta information about inference
    txt_file_path = os.path.join(output_dir, f'{output_dir}-metainfo.txt')
    with open(txt_file_path, "w") as file:
        print(f"Metainfo txt file is saving to {output_dir}", flush=True)
        file.write(f"{output_dir=}\n"
                   f"{args.dataset_dir=}\n"
                   f"{args.models_dir=}\n"
                   f"{args.batch_size=}\n"
                   f"{image_size=}\n"
                   f"{device=}\n"
                   f"{time_total=}\n"
                   f"sample_size={sample_size}\n"
                   f"total_memory_size={total_memory_size}\n")


if __name__ == '__main__':
    start_time = time.time()
    args = parse_arguments()
    image_size = f"{args.image_width}, {args.image_height}"
    image_size = eval(image_size)
    print(f"image_size: {image_size}", flush=True)
    image_dict = preload_image_names(args.dataset_dir)
    data_loader = generate_dataset(args.dataset_dir, image_size, args.batch_size)
    sample_size, total_memory_size = calculate_memory_and_sample_size(image_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_predictions(args.models_dir, data_loader, args.output_dir, image_size, image_dict)
    print("All done :)", flush=True)
    get_metainfo(start_time, args.output_dir, sample_size, total_memory_size)
