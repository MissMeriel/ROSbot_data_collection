import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='parent directory of training dataset')
    args = parser.parse_args()
    return args

def count_samples(args):
    directory = args.dir #"/p/rosbot/rosbotxl/data-meriel/cleaned/" "/p/rosbot/rosbotxl/data-all/data-yili-cleaned/"
    collections = [i for i in os.listdir(directory) if 'collection' in i]
    print(f"Total collections: {len(collections)}")
    total_samples = 0 # dataset sample counter
    total_size = 0 # size on disk
    for i in collections:
        count = [j for j in os.listdir(directory + i) if "jpg" in j]
        total_samples += len(count)
        size = sum(os.path.getsize(directory + i +"/"+ f) for f in os.listdir(directory + i) if os.path.isfile(directory + i + "/"+ f))
        total_size += size
    print(f"Total samples in {directory}: {total_samples} \n{total_samples/5:.1f} seconds\n{(total_size/1e6):.1f} Mb")

if __name__ == '__main__':
    args= parse_args()
    count_samples(args)