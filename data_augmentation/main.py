# main.py

import argparse
from processing import process_parent_dir

def parse_args():
    """
    Parse command line arguments to configure the script's behavior.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--parentdir", type=str, default='/home/husarion/media/usb/rosbotxl_data')
    parser.add_argument("--img_filename_key", type=str, default="image name")
    parser.add_argument("--level", type=str, choices=['rosbotxl_data', 'collection'], default='rosbotxl_data',
                        help="Specify the directory level to process: 'rosbotxl_data' for the whole dataset or 'collection' for a single collection.")
    parser.add_argument("--transformations", type=str, nargs='+', help="List of transformations to apply. Example: --transformations blur contrast horizontal_flip")
    parser.add_argument("--composed_transforms", type=str, nargs='+', help="List of composed transformations to apply. Example: --composed_transforms blur,contrast random_crop,brightness")
    parser.add_argument("--specify", type=str, nargs='+', help="List of specific image paths to process.")
    args = parser.parse_args()
    return args

def main():
    """
    Main function to process images based on provided command line arguments.
    """
    args = parse_args()
    composed_transforms = [ct.split(',') for ct in args.composed_transforms] if args.composed_transforms else None
    specified_images = args.specify if args.specify else None
    process_parent_dir(args.parentdir, args.level, args.img_filename_key, composed_transforms, specified_images)

if __name__ == '__main__':
    main()
