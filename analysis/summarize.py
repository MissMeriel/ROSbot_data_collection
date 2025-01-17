import os
import argparse
import pandas as pd
import numpy as np
import cv2

'''
The summarize  functionality enables trace summarization at various levels -- by model, attributes, or trace length -- to better group these traces and understand the behaviors exhibited by models under deployment. 
summarize  takes several flags: -mfdvat .
The -mf  flags function like those in query  to specify model and failures.
-d, --mode  specifies one of several modes: statistical , temporal ,  visual , and  conditional .
statistical  mode summarizes the failure trace by mean, median, mode, standard deviation and variance, and range of each of the trace variables.
temporal  mode produces the statistical metrics in 5 second intervals.
visual  mode visualizes each numerical trace variable as a histogram and boxplot.
conditional mode produces the statistical metrics conditioned upon the values of the variable -v, --target_variable  flag.
Each mode can use the -a, --aggregate  flag which takes the entire trace into consideration, or in slices specified by -t, --timestep . 
They can also be run at the model level or summarize individual failures in aggregate. 
The summarize  functionality can help to compare and differentiate traces, assisting with failure clustering and fault isolation.
'''

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", type=str, help='model ID')
    parser.add_argument('-f', "--failure", type=str, help='failure ID (can be in the form of F# or F#,F#,F#...)')
    parser.add_argument('-d', "--mode", type=str, choices=['statistical', 'temporal',  'visual', 'conditional'], 
                            help='analysis mode with which to summarize the trace.\
                                statistical  mode summarizes the failure trace by mean, median, mode, standard deviation and variance, and range of each of the trace variables.\
                                temporal  mode produces the statistical metrics in 5 second intervals.\
                                visual  mode visualizes each numerical trace variable as a histogram and boxplot.\
                                conditional mode produces the statistical metrics conditioned upon the values of the variable -v, --target_variable  flag.')
    parser.add_argument('-o', "--outdir", type=str, default="./summarize-output/", help='directory in which to save output')
    parser.add_argument('-v', '--target_variable', type=str, choices=['image name','linear_speed_x','angular_speed_z','is_turning','is_manually_off_course','lidar_ranges'])
    parser.add_argument('-l', '--training_metas', action="store_true", help="print out the training information for the model specified in -m,--model")
    parser.add_argument("-a", "--aggregate", action="store_true", help='takes the entire trace into consideration')
    parser.add_argument("-t", "--timestep", type=str, default="./summarize-output/", help='directory in which to save output')
    parser.add_argument("--resize", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    return args

# Moments are 1=mean 2=variance 3=skewness, 4=kurtosis
def get_distribution_moments(arr):
    moments = {}
    moments['shape'] = np.asarray(arr).shape
    moments['mean'] = np.mean(arr)
    moments['median'] = np.median(arr)
    moments['var'] = np.var(arr)
    moments['skew'] = stats.skew(arr)
    moments['kurtosis'] = stats.kurtosis(arr)
    moments['max'] = max(arr)
    moments['min'] = min(arr)
    return moments


def statistical(df):
    print(df.columns)
    print("angular_speed_z:")
    turning_mean = df.loc[:, 'angular_speed_z'].mean() 
    turning_var = df.loc[:, 'angular_speed_z'].var() 
    print(turning_mean, turning_var, df.shape[0])
    print("lidar_ranges:")
    lidar_ranges_raw = [] #df.loc[:, 'lidar_ranges']
    for index, row  in df.iterrows():
        r = row['lidar_ranges']
        r_floats = [float(i) for i in r.split(" ")]
        # r_floats_filtered = 
        r_floats = filter(lambda x: x != float('-inf'), r_floats)
        # print(r_floats_filtered)
        print("r_floats", r_floats)
        lidar_ranges_raw.append(r_floats_filtered)

    lidar_mean = np.mean(lidar_ranges_raw)
    lidar_var = np.var(lidar_ranges_raw)
    print(lidar_mean, lidar_var)
    # lidar_var = df.loc[:, 'lidar_ranges'].var()
    # lidar_min = df.loc[:, 'lidar_ranges'].min()
    # lidar_max = df.loc[:, 'lidar_ranges'].max()
    # print(lidar_mean, lidar_var, lidar_min, lidar_max)
    return turning_mean, turning_var


def main(args):
    d = f"../failure-catalog/{args.model}/"
    if args.failure:
        f = args.failure.split(",")
        ds = [d+i+"/" for i in f]
    else:
        ds = [d + i for i in os.listdir(d) if os.path.isdir(d+i)]
    print(ds)
    dfs = []
    for fd in ds:
        csvfile = fd + "/data_cleaned.csv"
        df = pd.read_csv(csvfile)
        
        dfs.append(df)
    concat_df = pd.concat(dfs)
    if args.mode == "statistical":
        results = statistical(concat_df)

    if args.training_metas:
        modelsubdir = ["../pretrained-models/" + i for i in os.listdir("../pretrained-models") if args.model+"-" in i][0]
        modeldir = f"../pretrained-models/{args.model}/"
        a = [modeldir+ i for i in os.listdir(modeldir) if "metainfo.txt" in i][0]
        print(a)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)