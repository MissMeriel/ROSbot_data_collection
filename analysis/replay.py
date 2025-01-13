from PIL import Image
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='parent directory of training dataset')
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--resize", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    
    pass

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

