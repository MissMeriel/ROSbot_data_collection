import numpy as np
import sys
import shutil
import os, cv2, csv
import PIL
from PIL import Image
import copy
from scipy import stats
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import skimage
import skimage.io as sio
import argparse

# Parse command line args into 'args' variable
# To use command line args: python3 clean_data.py --parentdir /u/<id>/ --level collection
parser = argparse.ArgumentParser()
parser.add_argument("--parentdir", type=str, default='/home/husarion/media/usb/rosbotxl_data')
parser.add_argument("--img_filename_key", type=str, default="image name")
parser.add_argument("--level", type=str, choices=['rosbotxl_data', 'collection'], default='rosbotxl_data', 
                    help="Specify the directory level to process: 'rosbotxl_data' for the whole dataset or 'collection' for a single collection.")
args = parser.parse_args()
print("args:" + str(args))

parsable_columns = [""]


def data_analysis_in_collection(collection_dir):
    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        # read in csv
        try:
            df = pd.read_csv(f"{collection_dir}/data.csv")
        except FileNotFoundError as e:
            try:
                df = pd.read_csv(f"{collection_dir}/data.txt")
            except FileNotFoundError as e:
                print(e, "\nNo data.csv or data.txt in directory")
                return
        print(f"{df.columns=}")
        # iterate over columns in csv
        for col in df.columns:
            # get all the values for a column
            column_values = df[col].to_numpy()
            print(f"\n{col} {df[col].apply(type)[0]}")
            print(f"{col} {column_values.dtype=}")
            if df[col].apply(type)[0] is str:
                print(f"{col} is a string")
            # if column is a numpy numeric subdtype
            elif np.issubdtype(column_values.dtype, np.floating) or np.issubdtype(column_values.dtype, np.integer):
                # take the average of the values and print it out
                avg = np.average(column_values)
                print(f"{col} Average: {avg}")
    else:
        print(f"Invalid directory: {collection_dir}")


def data_analysis(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                data_analysis_in_collection(p)
    elif level == 'collection':
        data_analysis_in_collection(Path(parentdir))


def process_dirs_in_collection(collection_dir, img_filename_key="image name"):
    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        # read in csv
        try:
            df = pd.read_csv(f"{collection_dir}/data.csv")
        except FileNotFoundError as e:
            try:
                df = pd.read_csv(f"{collection_dir}/data.txt")
            except FileNotFoundError as e:
                print(e, "\nNo data.csv or data.txt in directory")
                return
        print(f"{df.columns=}")
        print(f"Data frame loaded from {collection_dir}: \n{df}")
        # iterate over files inside of directory collection_dir
        for pp in Path(collection_dir).iterdir():
            # check if current file pp has an image file extension
            if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                print(f"Looking for {pp} inside dataframe")
                # find the index of the row that this image appears inside data.csv
                df_index = df.index[df[img_filename_key] == pp.name]
                print(f"{df_index=} \n{df.loc[df_index]}")
                # find the steering value in that row using the column name in img_filename_key
                orig_y_steer = df.loc[df_index, img_filename_key].item()
                print(f"{orig_y_steer=}\n")


def process_dirs(parentdir, level, img_filename_key="image name"):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                process_dirs_in_collection(p, img_filename_key)
    elif level == 'collection':
        process_dirs_in_collection(Path(parentdir), img_filename_key)


def clean_corrupted_images_in_collection(collection_dir):
    # the directory to move corrupted images
    corrupted_images_dir = Path(collection_dir) / "corrupted_images"
    corrupted_images_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
    
    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        # read in csv
        try:
            df = pd.read_csv(f"{collection_dir}/data.csv")
        except FileNotFoundError as e:
            try:
                df = pd.read_csv(f"{collection_dir}/data.txt")
            except FileNotFoundError as e:
                print(e, "\nNo data.csv or data.txt in directory")
                return
        
        deleted_items = pd.DataFrame(columns=df.columns)
        # iterate over files inside of directory collection_dir
        for pp in Path(collection_dir).iterdir():
            # check if current file pp has an image file extension
            if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                print(f"Verifying {pp}")
                try:
                    # verify with PIL
                    im = Image.open(pp)
                    im.verify()
                except PIL.UnidentifiedImageError as e:
                    # move the corrupted image to the new directory
                    print(e, f"\nMoving {pp} to {corrupted_images_dir}")
                    shutil.move(pp, corrupted_images_dir / pp.name)
                    # add row to deleted_items
                    df_index = df.index[df[args.img_filename_key] == pp.name]
                    deleted_items = pd.concat([deleted_items, df.loc[df_index]])
                    df = df.drop(df_index)
        
        # Load the deletedItem.csv file if it exists
        deleted_items_file = collection_dir / "deletedItem.csv"
        if deleted_items_file.exists():
            deleted_items_df = pd.read_csv(deleted_items_file)
            # Exclude rows with corrupted images from data_cleaned.csv
            df = df[~df['image name'].isin(deleted_items_df['image name'])]
            
        # Save the updated data frame and the deleted items
        df.to_csv(f"{collection_dir}/data_cleaned.csv", mode='w', index=False)
        deleted_items.to_csv(f"{collection_dir}/deletedItem.csv", mode='a', index=False, header=not os.path.exists(f"{collection_dir}/deletedItem.csv"))

        

def clean_corrupted_images(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                clean_corrupted_images_in_collection(p)
    elif level == 'collection':
        clean_corrupted_images_in_collection(Path(parentdir))


def clean_data_in_collection(collection_dir):
    # the directory to move the corresponding image files of the data that needs to be cleaned out
    removed_images_dir = collection_dir / "removed_images"
    removed_images_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
    
    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        print(f"Running clean_data on {collection_dir}")
        # read in csv
        try:
            df = pd.read_csv(f"{collection_dir}/data.csv")
        except FileNotFoundError as e:
            try:
                df = pd.read_csv(f"{collection_dir}/data.txt")
            except FileNotFoundError as e:
                print(e, "\nNo data.csv or data.txt in directory")
                return
        print(f"Original data frame loaded from {collection_dir}: \n{df}")
        
        # Load the deletedItem.csv file if it exists
        deleted_items_file = collection_dir / "deletedItem.csv"
        if deleted_items_file.exists():
            deleted_items_df = pd.read_csv(deleted_items_file)
            # Exclude rows with corrupted images from data_cleaned.csv
            df = df[~df['image name'].isin(deleted_items_df['image name'])]
        
        # Filter out rows where linear_speed_x == 0.0, angular_speed_z == 0.0, and is_turning == False
        condition = (df['linear_speed_x'] != 0.0) | (df['angular_speed_z'] != 0.0) | (df['is_turning'] == True)
        df_cleaned = df[condition]
        df_deletedItem = df[~condition]
        print(f"Cleaned data frame: \n{df_cleaned}")
        
        # Save the cleaned data back to the file
        df_cleaned.to_csv(f"{collection_dir}/data_cleaned.csv", mode='w', index=False)
        print(f"Cleaned data saved to {collection_dir}/data_cleaned.csv")
        df_deletedItem.to_csv(f"{collection_dir}/deletedItem.csv", mode='a', index=False, header=not os.path.exists(f"{collection_dir}/deletedItem.csv"))
        print(f"Cleaned data saved to {collection_dir}/data_cleaned.csv")
        
        # Move corresponding image files of the removed data to a new directory
        for index, row in df_deletedItem.iterrows():
            img_filename = row['image name']  # Assuming the image filename is stored in a column named 'image name'
            img_path = collection_dir / img_filename
            if img_path.exists():
                shutil.move(img_path, removed_images_dir / img_path.name)
                print(f"Moved {img_path} to {removed_images_dir / img_path.name}")
    else:
        print(f"Invalid directory: {collection_dir}")


def clean_data(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                clean_data_in_collection(p)
    elif level == 'collection':
        clean_data_in_collection(Path(parentdir))


def clean_removed_image_dirs_in_collection(collection_dir):
    removed_images_dir = Path(collection_dir) / "removed_images"
    corrupted_images_dir = Path(collection_dir) / "corrupted_images"
    
    # remove cleaned_images directory
    if removed_images_dir.exists() and removed_images_dir.is_dir():
        shutil.rmtree(removed_images_dir)
        print(f"Directory {removed_images_dir} has been deleted.")
    else:
        print(f"No removed_images directory found in {collection_dir}.")
    
    # remove corrupted_images directory
    if corrupted_images_dir.exists() and corrupted_images_dir.is_dir():
        shutil.rmtree(corrupted_images_dir)
        print(f"Directory {corrupted_images_dir} has been deleted.")
    else:
        print(f"No corrupted_images directory found in {collection_dir}.")
    cleaned_images_dir = Path(collection_dir) / "cleaned_images"
    if cleaned_images_dir.exists() and cleaned_images_dir.is_dir():
        print(f"Deleting directory and its contents: {cleaned_images_dir}")
        shutil.rmtree(cleaned_images_dir)
        print(f"Directory {cleaned_images_dir} has been deleted.")
    else:
        print(f"No cleaned_images directory found in {collection_dir}.")


def clean_removed_image_dirs(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                clean_removed_image_dirs_in_collection(p)
    elif level == 'collection':
        clean_removed_image_dirs_in_collection(Path(parentdir))
        
# Runs the code inside the if statement when the program is run directly by the Python interpreter
# e.g. when you call this script from the command line like: python3 clean_rosbot_data.py --parentdir /p/rosbot/rosbot_dataset --level collection
if __name__ == '__main__':
    # ONLY RUN clean_removed_image_dirs() IF YOU ARE DELETING IMAGES, OTHERWISE COMMENT THE LINE BELOW
    # clean_corrupted_images(args.parentdir, args.level)
    # process_dirs(args.parentdir, args.level, args.img_filename_key)
    # data_analysis(args.parentdir, args.level)
    # clean_data(args.parentdir, args.level)
    # clean_removed_image_dirs(args.parentdir, args.level)
