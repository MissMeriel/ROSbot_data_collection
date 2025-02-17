from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import argparse
import pandas as pd
import example_monitors
import cv2
import numpy as np

'''
The replay functionality enables running a trace step by step until the failure. 
replay takes in the model ID, failure ID, and replay speed, and replays a trace with the designated replay speed factor.
This replay also includes a plug-in where the user can to connect monitoring code through analysis/example_monitors.py, 
e.g. python3 replay.py --monitor example_monitors.detectLidarProximity <modelIDfailureID>.
'''

def parse_arguments():
    parser = argparse.ArgumentParser(prog='replay',
                    description='The replay functionality enables running a trace step by step until the failure.',
                    epilog='replay also includes a plug-in where the user can to connect monitoring code through analysis/example_monitors.py, e.g. python3 replay.py --monitor example_monitors.detectLidarProximity <modelIDfailureID>.')
    parser.add_argument('-m', "--model", type=str, help='model ID')
    parser.add_argument('-f', "--failure", type=str, help='failure ID')
    parser.add_argument('-o', "--outdir", type=str, default="replay-output/", help='directory in which to save output')
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--resize", type=str, default=None)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    return args

def add_text(img, text):
    img2 = Image.new('RGB', (int(img.size[0]*3/2), int(img.size[1])))
    img2.paste(img, (0,0))
    d = ImageDraw.Draw(img2)
    font = ImageFont.truetype("figures/IBMPlexMono-Bold.ttf", 35)
    if "True" in text:
        d.text((img.size[0]+20, 20), text, fill=(255, 0, 0), font=font)
    else:
        d.text((img.size[0]+20, 20), text, fill=(255, 255, 255), font=font)
    return img2

def sorter(x):
    return int(x.replace(".jpg", "").split("-")[-1])

def main(args):
    d = f"../failure-catalog/{args.model}/{args.failure}/"
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    images = [d+i for i in os.listdir(d) if "jpg" in i]
    images = sorted(images, key=sorter)
    df = pd.read_csv(d + "data_cleaned.csv")
    df = df.reset_index()
    print(df.columns)
    monitor_output = None
    for index, row in df.iterrows():
        print(row['image_name'])
        lidar_ranges = [float(i) for i in row["lidar_ranges"].split(" ")]
        print(min(lidar_ranges), max(lidar_ranges))
        img = Image.open(d + row['image_name'])
        img = img.resize((int(img.size[0]/2), int(img.size[1]/2)), Image.Resampling.LANCZOS)
        if args.monitor:
            monitor = getattr(example_monitors, args.monitor)
            monitor_output = monitor(row)
        text = f"{args.model} {args.failure} \n{row['image_name']}\ntimestamp={row['timestamp']:.1f}\nlinear_speed_x={row['linear_speed_x']:.3f}\nangular_speed_z={row['angular_speed_z']:.3f}\nmonitor={args.monitor}\nmonitor_output={monitor_output}"
        img = add_text(img, text)
        if args.save:
            print(f"Saving image to {outdir}...")
            img.save(f"{outdir}/{row['image_name']}")
        if args.show:
            open_cv_image = np.array(img)
            cv2.imshow('replay', open_cv_image)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == '__main__':

    args = parse_arguments()
    main(args)

