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
parser.add_argument("--start", type=str, help="Start image filename for cleaning data.")
parser.add_argument("--end", type=str, help="End image filename for cleaning data.")
parser.add_argument("--startTime", type=int, help="Start time for plot.")
parser.add_argument("--endTime", type=int, help="End time for plot.")
args = parser.parse_args()
print("args:" + str(args))

parsable_columns = [""]


def data_analysis_in_collection(collection_dir, file_name):
    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        # read in csv
        try:
            df = pd.read_csv(f"{collection_dir}/{file_name}")
        except FileNotFoundError as e:
            print(e, f"\nNo {file_name} in directory")
            return
        print(f"Analysis for {collection_dir} using {file_name}")
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
        aggregated_df_original = []
        aggregated_df_cleaned = []
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                data_analysis_in_collection(p, "data.csv")
                data_analysis_in_collection(p, "data_cleaned.csv")

                try:
                    df_original = pd.read_csv(f"{p}/data.csv")
                    aggregated_df_original.append(df_original)
                except FileNotFoundError:
                    pass

                try:
                    df_cleaned = pd.read_csv(f"{p}/data_cleaned.csv")
                    aggregated_df_cleaned.append(df_cleaned)
                except FileNotFoundError:
                    pass

        if aggregated_df_original:
            combined_df_original = pd.concat(aggregated_df_original, ignore_index=True)
            print("Combined Analysis for all collections (Original Data)")
            print(f"{combined_df_original.columns=}")
            for col in combined_df_original.columns:
                column_values = combined_df_original[col].to_numpy()
                print(f"\n{col} {combined_df_original[col].apply(type)[0]}")
                print(f"{col} {column_values.dtype=}")
                if combined_df_original[col].apply(type)[0] is str:
                    print(f"{col} is a string")
                elif np.issubdtype(column_values.dtype, np.floating) or np.issubdtype(column_values.dtype, np.integer):
                    avg = np.average(column_values)
                    print(f"{col} Average: {avg}")

        if aggregated_df_cleaned:
            combined_df_cleaned = pd.concat(aggregated_df_cleaned, ignore_index=True)
            print("Combined Analysis for all collections (Cleaned Data)")
            print(f"{combined_df_cleaned.columns=}")
            for col in combined_df_cleaned.columns:
                column_values = combined_df_cleaned[col].to_numpy()
                print(f"\n{col} {combined_df_cleaned[col].apply(type)[0]}")
                print(f"{col} {column_values.dtype=}")
                if combined_df_cleaned[col].apply(type)[0] is str:
                    print(f"{col} is a string")
                elif np.issubdtype(column_values.dtype, np.floating) or np.issubdtype(column_values.dtype, np.integer):
                    avg = np.average(column_values)
                    print(f"{col} Average: {avg}")

    elif level == 'collection':
        data_analysis_in_collection(Path(parentdir), "data.csv")
        data_analysis_in_collection(Path(parentdir), "data_cleaned.csv")


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

        # Exclude rows from deletedItem.csv, data_range_cleaned.csv, and Not_recentering_correction.csv
        df = exclude_data_from_files(df, collection_dir)

        # Save the updated data frame and the deleted items
        df.to_csv(f"{collection_dir}/data_cleaned.csv", mode='w', index=False)
        deleted_items.to_csv(f"{collection_dir}/deletedItem.csv", mode='a', index=False,
                             header=not os.path.exists(f"{collection_dir}/deletedItem.csv"))


def clean_corrupted_images(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                clean_corrupted_images_in_collection(p)
    elif level == 'collection':
        clean_corrupted_images_in_collection(Path(parentdir))


def clean_data_resting_in_collection(collection_dir):
    # the directory to move the corresponding image files of the data that needs to be cleaned out
    removed_images_dir = collection_dir / "removed_images"
    removed_images_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        print(f"Running clean_data_resting on {collection_dir}")
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

        # Exclude rows from deletedItem.csv, data_range_cleaned.csv, and Not_recentering_correction.csv
        df = exclude_data_from_files(df, collection_dir)

        # Filter out rows where linear_speed_x == 0.0, angular_speed_z == 0.0, and is_turning == False
        condition = (df['linear_speed_x'] != 0.0) | (df['angular_speed_z'] != 0.0) | (df['is_turning'] == True)
        df_cleaned = df[condition]
        df_deletedItem = df[~condition]
        print(f"Cleaned data frame: \n{df_cleaned}")

        # Save the cleaned data back to the file
        df_cleaned.to_csv(f"{collection_dir}/data_cleaned.csv", mode='w', index=False)
        print(f"Cleaned data saved to {collection_dir}/data_cleaned.csv")
        df_deletedItem.to_csv(f"{collection_dir}/deletedItem.csv", mode='a', index=False,
                              header=not os.path.exists(f"{collection_dir}/deletedItem.csv"))
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


def clean_data_resting(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                clean_data_resting_in_collection(p)
    elif level == 'collection':
        clean_data_resting_in_collection(Path(parentdir))


def clean_removed_image_dirs_in_collection(collection_dir):
    removed_images_dir = Path(collection_dir) / "removed_images"
    corrupted_images_dir = Path(collection_dir) / "corrupted_images"
    range_removed_images_dir = Path(collection_dir) / "range_removed_images"
    not_recentering_correction_dir = Path(collection_dir) / "Not_recentering_correction"

    # Remove removed_images directory
    if removed_images_dir.exists() and removed_images_dir.is_dir():
        shutil.rmtree(removed_images_dir)
        print(f"Directory {removed_images_dir} has been deleted.")
    else:
        print(f"No removed_images directory found in {collection_dir}.")

    # Remove corrupted_images directory
    if corrupted_images_dir.exists() and corrupted_images_dir.is_dir():
        shutil.rmtree(corrupted_images_dir)
        print(f"Directory {corrupted_images_dir} has been deleted.")
    else:
        print(f"No corrupted_images directory found in {collection_dir}.")

    # Remove range_removed_images directory
    if range_removed_images_dir.exists() and range_removed_images_dir.is_dir():
        shutil.rmtree(range_removed_images_dir)
        print(f"Directory {range_removed_images_dir} has been deleted.")
    else:
        print(f"No range_removed_images directory found in {collection_dir}.")

    # Remove Not_recentering_correction directory
    if not_recentering_correction_dir.exists() and not_recentering_correction_dir.is_dir():
        shutil.rmtree(not_recentering_correction_dir)
        print(f"Directory {not_recentering_correction_dir} has been deleted.")
    else:
        print(f"No Not_recentering_correction directory found in {collection_dir}.")


def clean_removed_image_dirs(parentdir, level):
    if level == 'rosbotxl_data':
        # Iterate over each collection directory
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                clean_removed_image_dirs_in_collection(p)
    elif level == 'collection':
        clean_removed_image_dirs_in_collection(Path(parentdir))


def plot_distribution_in_collection(collection_dir):
    angular_speed_data_original = []
    angular_speed_data_cleaned = []

    if collection_dir is not None and collection_dir.is_dir():
        try:
            df = pd.read_csv(f"{collection_dir}/data.csv")
            angular_speed_data_original.extend(df['angular_speed_z'].tolist())
        except FileNotFoundError as e:
            print(e, "\nNo data.csv in directory")

        try:
            df_cleaned = pd.read_csv(f"{collection_dir}/data_cleaned.csv")
            angular_speed_data_cleaned.extend(df_cleaned['angular_speed_z'].tolist())
        except FileNotFoundError as e:
            print(e, "\nNo data_cleaned.csv in directory")

    return angular_speed_data_original, angular_speed_data_cleaned


def plot_distribution_original_data(parentdir, level):
    aggregated_angular_speed_data_original = []
    aggregated_angular_speed_data_cleaned = []

    if level == 'rosbotxl_data':
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                data_original, data_cleaned = plot_distribution_in_collection(p)
                aggregated_angular_speed_data_original.extend(data_original)
                aggregated_angular_speed_data_cleaned.extend(data_cleaned)

        plt.figure(figsize=(10, 6))
        plt.hist(aggregated_angular_speed_data_original, bins=30, edgecolor='black')
        plt.title(f'Distribution of angular_speed_z in all collections (Original Data) - Level: {level}')
        plt.xlabel('angular_speed_z')
        plt.ylabel('Frequency')
        # plt.yscale("log")
        plt.grid(True)
        plt.show()


    elif level == 'collection':
        data_original, data_cleaned = plot_distribution_in_collection(Path(parentdir))

        plt.figure(figsize=(10, 6))
        plt.hist(data_original, bins=30, edgecolor='black')
        plt.title(f'Distribution of angular_speed_z in {parentdir} (Original Data) - Level: {level}')
        plt.xlabel('angular_speed_z')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def plot_distribution_cleaned_data(parentdir, level):
    aggregated_angular_speed_data_original = []
    aggregated_angular_speed_data_cleaned = []

    if level == 'rosbotxl_data':
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                data_original, data_cleaned = plot_distribution_in_collection(p)
                aggregated_angular_speed_data_original.extend(data_original)
                aggregated_angular_speed_data_cleaned.extend(data_cleaned)

        plt.figure(figsize=(10, 6))
        plt.hist(aggregated_angular_speed_data_cleaned, bins=30, edgecolor='black')
        plt.title(f'Distribution of angular_speed_z in all collections (Cleaned Data) - Level: {level}')
        plt.xlabel('angular_speed_z')
        plt.ylabel('Frequency')
        # plt.yscale("log")
        plt.grid(True)
        plt.show()

    elif level == 'collection':
        data_original, data_cleaned = plot_distribution_in_collection(Path(parentdir))

        plt.figure(figsize=(10, 6))
        plt.hist(data_cleaned, bins=30, edgecolor='black')
        plt.title(f'Distribution of angular_speed_z in {parentdir} (Cleaned Data) - Level: {level}')
        plt.xlabel('angular_speed_z')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


def clean_data_range_in_collection(collection_dir, start_img, end_img):
    # the directory to move the corresponding image files of the data that needs to be cleaned out
    range_removed_images_dir = collection_dir / "range_removed_images"
    range_removed_images_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    # check to make sure you passed in a valid directory
    if collection_dir is not None and collection_dir.is_dir():
        print(f"Running clean_data_range on {collection_dir} from {start_img} to {end_img}")
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

        # Exclude rows from deletedItem.csv, data_range_cleaned.csv, and Not_recentering_correction.csv
        df = exclude_data_from_files(df, collection_dir)

        # Filter out rows within the specified range
        condition = (df['image name'] >= start_img) & (df['image name'] <= end_img)
        df_range = df[condition]
        df_cleaned = df[~condition]
        print(f"Data in range: \n{df_range}")

        # Save the cleaned data back to the file
        df_cleaned.to_csv(f"{collection_dir}/data_cleaned.csv", mode='w', index=False)
        print(f"Cleaned data saved to {collection_dir}/data_cleaned.csv")
        df_range.to_csv(f"{collection_dir}/data_range_cleaned.csv", mode='a', index=False,
                        header=not os.path.exists(f"{collection_dir}/data_range_cleaned.csv"))
        print(f"Cleaned data saved to {collection_dir}/data_range_cleaned.csv")

        # Move corresponding image files of the removed data to a new directory
        for index, row in df_range.iterrows():
            img_filename = row['image name']  # Assuming the image filename is stored in a column named 'image name'
            img_path = collection_dir / img_filename
            if img_path.exists():
                shutil.move(img_path, range_removed_images_dir / img_path.name)
                print(f"Moved {img_path} to {range_removed_images_dir / img_path.name}")
    else:
        print(f"Invalid directory: {collection_dir}")


def only_keep_recentering_correction_in_collection(collection_dir):
    not_recentering_correction_dir = collection_dir / "Not_recentering_correction"
    not_recentering_correction_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

    not_recentering_correction_file = collection_dir / "Not_recentering_correction.csv"

    if collection_dir.is_dir():
        try:
            df = pd.read_csv(f"{collection_dir}/data.csv")
        except FileNotFoundError as e:
            print(e, "\nNo data.csv in directory")
            return

        # Filter the data
        condition = (df['is_manually_off_course'] == True) | (df['angular_speed_z'] == 0.0)
        df_not_recentering_correction = df[condition]

        # Save the filtered data to Not_recentering_correction.csv
        df_not_recentering_correction.to_csv(not_recentering_correction_file, mode='w', index=False, header=True)
        print(f"Saved not recentering correction data to {not_recentering_correction_file}")

        # Move corresponding image files of the removed data to a new directory
        for index, row in df_not_recentering_correction.iterrows():
            img_filename = row['image name']
            img_path = collection_dir / img_filename
            if img_path.exists():
                shutil.move(img_path, not_recentering_correction_dir / img_path.name)
                print(f"Moved {img_path} to {not_recentering_correction_dir / img_path.name}")

        # If data_cleaned.csv exists, update it
        data_cleaned_file = collection_dir / "data_cleaned.csv"
        if data_cleaned_file.exists():
            df_cleaned = pd.read_csv(data_cleaned_file)
            df_cleaned = df_cleaned[~df_cleaned['image name'].isin(df_not_recentering_correction['image name'])]
            df_cleaned.to_csv(data_cleaned_file, mode='w', index=False)
            print(f"Updated data_cleaned.csv in {collection_dir}")
        else:
            # Create data_cleaned.csv with remaining data
            df_remaining = df[~df['image name'].isin(df_not_recentering_correction['image name'])]
            df_remaining.to_csv(data_cleaned_file, mode='w', index=False)
            print(f"Created data_cleaned.csv in {collection_dir}")
    else:
        print(f"Invalid directory: {collection_dir}")


def only_keep_recentering_correction(parentdir, level):
    if level == 'rosbotxl_data':
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                only_keep_recentering_correction_in_collection(p)
    elif level == 'collection':
        only_keep_recentering_correction_in_collection(Path(parentdir))


def plot_steering_overtime_cleaned_data(parentdir, level, start=None, end=None):
    aggregated_angular_speed_data_cleaned = []

    if level == 'rosbotxl_data':
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                data_original, data_cleaned = plot_distribution_in_collection(p)
                aggregated_angular_speed_data_cleaned.extend(data_cleaned)

        if start is not None and end is not None:
            aggregated_angular_speed_data_cleaned = aggregated_angular_speed_data_cleaned[start:end]

        plt.figure(figsize=(10, 6))
        plt.plot(aggregated_angular_speed_data_cleaned)
        plt.title(f'Angular Speed over Time in all collections (Cleaned Data) - Level: {level}')
        plt.xlabel('Time')
        plt.ylabel('Angular Speed')
        plt.grid(True)
        plt.show()

    elif level == 'collection':
        data_original, data_cleaned = plot_distribution_in_collection(Path(parentdir))

        if start is not None and end is not None:
            data_cleaned = data_cleaned[start:end]

        plt.figure(figsize=(10, 6))
        plt.plot(data_cleaned)
        plt.title(f'Angular Speed over Time in {parentdir} (Cleaned Data) - Level: {level}')
        plt.xlabel('Time')
        plt.ylabel('Angular Speed')
        plt.grid(True)
        plt.show()


def exclude_data_from_files(df, collection_dir):
    """
    Exclude rows from the DataFrame based on the 'image name' column
    if they are listed in certain files within the collection directory.
    """
    files_to_check = ["deletedItem.csv", "data_range_cleaned.csv", "Not_recentering_correction.csv"]
    for file_name in files_to_check:
        file_path = collection_dir / file_name
        if file_path.exists():
            items_df = pd.read_csv(file_path)
            df = df[~df['image name'].isin(items_df['image name'])]
    return df


# Runs the code inside the if statement when the program is run directly by the Python interpreter
# e.g. when you call this script from the command line like: python3 clean_data.py --parentdir /p/rosbot/rosbot_dataset --level collection
if __name__ == '__main__':
# ONLY RUN clean_removed_image_dirs() IF YOU ARE DELETING IMAGES, OTHERWISE COMMENT THE LINE BELOW
# clean_corrupted_images(args.parentdir, args.level)
# process_dirs(args.parentdir, args.level, args.img_filename_key)
# clean_data_resting(args.parentdir, args.level)
# clean_data_range_in_collection(Path(args.parentdir), args.start, args.end)
# only_keep_recentering_correction(args.parentdir, args.level)
# clean_removed_image_dirs(args.parentdir, args.level)
# data_analysis(args.parentdir, args.level)
# plot_distribution_original_data(args.parentdir, args.level)
# plot_distribution_cleaned_data(args.parentdir, args.level)
# plot_steering_overtime_cleaned_data(args.parentdir, args.level, args.startTime, args.endTime)
