import time
import string
import random
import numpy as np
from scipy import stats

def randstr():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def timestr():
    localtime = time.localtime()
    return "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)


def pytorch_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.1f}MB'.format(size_all_mb))
    return size_all_mb

def characterize_steering_distribution(y_steering, generator):
    turning, straight = [], []
    for i in y_steering:
        if abs(i) < 0.1:
            straight.append(abs(i))
        else:
            turning.append(abs(i))
    # turning = [i for i in y_steering if i > 0.1]
    # straight = [i for i in y_steering if i <= 0.1]
    try:
        print("Moments of abs. val'd turning steering distribution:", generator.get_distribution_moments(turning))
        print("Moments of abs. val'd straight steering distribution:", generator.get_distribution_moments(straight))
    except Exception as e:
        print(e)
        print("len(turning)", len(turning))
        print("len(straight)", len(straight))

def get_outputs_distribution(arr: np.array):
    # all_outputs = np.array(all_outputs)
    moments = {}
    moments['shape'] = np.asarray(arr).shape
    moments['mean'] = np.mean(arr)
    moments['median'] = np.median(arr)
    moments['var'] = np.var(arr)
    moments['skew'] = stats.skew(arr)
    moments['kurtosis'] = stats.kurtosis(arr)
    moments['max'] = np.max(arr)
    moments['min'] = np.min(arr)
    return moments