# main.py

import argparse
from processing import process_parent_dir
from transformations import individual_transforms_with_level, individual_transformations_without_level

def parse_args():
    """
    Parse command line arguments to configure the script's behavior.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--parentdir", type=str, default='/home/husarion/media/usb/rosbotxl_data')
    parser.add_argument("--img_filename_key", type=str, default="image name")
    parser.add_argument("--level", type=str, choices=['rosbotxl_data', 'collection'], default='rosbotxl_data',
                        help="Specify the directory level to process: 'rosbotxl_data' for the whole dataset or 'collection' for a single collection.")
    parser.add_argument("-b", "--blur", type=float, default=0.0, help="Intensity level for blur effect, between 0.0 (no blur) and 1.0 (maximum blur).")
    parser.add_argument("-j", "--color_jitter", type=float, default=0.0, help="Intensity level for color jitter effect, between 0.0 (no effect) and 1.0 (maximum effect).")
    parser.add_argument("-c", "--random_crop", type=float, default=0.0, help="Intensity level for random crop, between 0.0 (no crop) and 1.0 (maximum crop size).")
    parser.add_argument("-n", "--brightness", type=float, default=1.0, help="Intensity level for brightness adjustment, between 0.0 (dark) and 2.0 (bright).")
    parser.add_argument("-t", "--contrast", type=float, default=1.0, help="Intensity level for contrast adjustment, between 0.0 (low contrast) and 2.0 (high contrast).")
    parser.add_argument("-s", "--shadow", type=float, default=0.0, help="Intensity level for adding shadows, between 0.0 (no shadow) and 1.0 (maximum shadow).")
    parser.add_argument("-d", "--time_of_day_dusk", type=float, default=0.0, help="Intensity level for dusk time of day effect, between 0.0 (day) and 1.0 (dusk).")
    parser.add_argument("-a", "--time_of_day_dawn", type=float, default=0.0, help="Intensity level for dawn time of day effect, between 0.0 (night) and 1.0 (dawn).")
    parser.add_argument("-e", "--elastic_transform", type=float, default=0.0, help="Intensity level for elastic transformation, between 0.0 (no distortion) and 1.0 (maximum distortion).")
    parser.add_argument("-l", "--lens_distortion", type=float, default=0.0, help="Intensity level for lens distortion effect, between 0.0 (no distortion) and 1.0 (noticeable distortion).")
    parser.add_argument("-o", "--noise", type=float, default=0.0, help="Intensity level for adding noise, between 0.0 (clean) and 1.0 (noisy).")
    parser.add_argument("-f", "--horizontal_flip", action='store_true', help="Apply horizontal flip to the images.")
    parser.add_argument("--composed_transforms", type=str, nargs='+', help="List of composed transformations to apply. Example: --composed_transforms blur,contrast random_crop,brightness")
    args = parser.parse_args()


    return args

def main():
    """
    Main function to process images based on provided command line arguments.
    """
    args = parse_args()
    composed_transforms = [ct.split(',') for ct in args.composed_transforms] if args.composed_transforms else None
    process_parent_dir(args.parentdir, args.level, args.img_filename_key, composed_transforms)

if __name__ == '__main__':
    main()
