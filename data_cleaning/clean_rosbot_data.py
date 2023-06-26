import numpy as np
import sys
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
# To use command line args: python3 clean_data.py --parentdir /u/<id>/
parser = argparse.ArgumentParser()
parser.add_argument("--parentdir",  type=str, default='/p/rosbot/rosbot_dataset')
parser.add_argument("--img_filename_key",  type=str, default="IMAGE")
args = parser.parse_args()
print("args:" + str(args))

parsable_columns = [""]

def data_analysis(parentdir):
    # check to make sure you passed in a valid directory
    if parentdir is not None and Path(parentdir).is_dir():
        # iterate over directories directly below parentdir
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                # read in csv
                try:
                    df = pd.read_csv(f"{p}/data.csv")
                except FileNotFoundError as e:
                    try:
                        df = pd.read_csv(f"{p}/data.txt")
                    except FileNotFoundError as e:
                        print(e, "\nNo data.csv or data.txt in directory")
                        exit(0)
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
                    # TODO: @Sam and @Zarif add to analysis here


def process_dirs(parentdir, img_filename_key="IMAGE"):
    # check to make sure you passed in a valid directory
    if parentdir is not None and Path(parentdir).is_dir():
        # iterate over directories directly below parentdir
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                # read in csv
                try:
                    df = pd.read_csv(f"{p}/data.csv")
                except FileNotFoundError as e:
                    try:
                        df = pd.read_csv(f"{p}/data.txt")
                    except FileNotFoundError as e:
                        print(e, "\nNo data.csv or data.txt in directory")
                        exit(0)
                print(f"{df.columns=}")
                print(f"Data frame loaded from {p}: \n{df}")
                # iterate over files inside of directory p
                for pp in Path(p).iterdir():
                    # check if current file pp has an image file extension
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                        print(f"Looking for {pp} inside dataframe")
                        # find the index of the row that this image appears inside data.csv
                        df_index = df.index[df[img_filename_key] == pp.name]
                        print(f"{df_index=} \n{df.loc[df_index]}")
                        # find the steering value in that row using the column name in img_filename_key
                        orig_y_steer = df.loc[df_index, img_filename_key].item()
                        print(f"{orig_y_steer=}\n")


def clean_corrupted_images(parentdir):
    # check to make sure you passed in a valid directory
    if parentdir is not None and Path(parentdir).is_dir():
        # iterate over directories directly below parentdir
        for p in Path(parentdir).iterdir():
            if p.is_dir():
                # iterate over files inside of directory p
                for pp in Path(p).iterdir():
                    # check if current file pp has an image file extension
                    if pp.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                        print(f"Verifying {pp}")
                        try:
                            # verify with PIL
                            im = Image.open(pp)
                            im.verify()
                        except PIL.UnidentifiedImageError as e:
                            # delete image if PIL cannot identify it
                            print(e, f"\nDeleting {pp}")
                            os.remove(pp)


# Runs the code inside the if statement when the program is run directly by the Python interpreter
# e.g. when you call this script from the command line like: python3 clean_rosbot_data.py --parentdir /p/rosbot/rosbot_dataset
if __name__ == '__main__':
    # ONLY RUN clean_corrupted_images() IF YOU ARE DELETING IMAGES, OTHERWISE COMMENT THE LINE BELOW
    # clean_corrupted_images(args.parentdir)
    # process_dirs(args.parentdir, args.img_filename_key)
    data_analysis(args.parentdir)