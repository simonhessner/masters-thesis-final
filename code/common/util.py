import json
import errno
import os
import datetime
import torch
from dsntnn import linear_expectation, normalized_linspace
import numpy as np
import torch.backends.cudnn as cudnn
import random


def count_parameters(model):
    """
    Return the (overall, trainable, fixed) parameters count

    :param model: PyTorch model
    :return: sum of parameter count, trainable parameters, fixed parameters
    """
    trainable = 0
    fixed = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            fixed += p.numel()
    return trainable+fixed, trainable, fixed


def mkdir_if_not_exists(directory, verbose=True):
    try:
        os.makedirs(directory)
        if verbose:
            print("Created folder '%s'" % directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def extract_key(dictionary, key):
    extracted = dictionary[key]
    rest = {k:v for k,v in dictionary.items() if k != key}
    return extracted, rest


def make_deterministic(random_seed=0):
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_human_readable_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_info(txt):
    print("[%s] %s" % (get_human_readable_time(), str(txt)))


def scale_coord(coord, width):
    """
    Converts coordinates from (-1,1) to (0, width)

    :param coord: n-dimensional coord where each dimension is in [-1,1]
    :param width: width of the output space per dimension (e.g. 128)
    :return: coord with each dimension in [0,width)
    """

    # In case of width=128 this is 63.5.
    # so the coord -1 would be mapped to 63.5 + (-1) * 63.5 = 0
    # and the coord 1 would be mapped to 63.5 + 1 * 63.5 = 127
    scale_factor = (width - 1) / 2

    return tuple(int(scale_factor + p * scale_factor) for p in coord) # from (-1,1) to (0,width-1)


class LossException(Exception):
    pass


def load_config(filename, verbose=True):
    if verbose:
        print("Loading config", filename)
    with open(filename, 'r') as f:
        config = json.load(f)
    if verbose:
        print(json.dumps(config, indent=1))
    config["config_file"] = filename
    return config


def move2device(tensor, gpu=None):
    if gpu is None or isinstance(gpu, int):
        dev = torch.device('cpu' if gpu in [None, -1] else 'cuda:%d' % int(gpu))
    elif isinstance(gpu, str):
        dev = torch.device(gpu)
    elif isinstance(gpu, torch.device):
        dev = gpu
    else:
        raise ValueError("Invalid location gpu=%s" % str(gpu))
    return tensor.to(dev)


def heatmap_variance(heatmaps):
    # copied and modified:
    # https://github.com/anibali/dsntnn/blob/4f20f5a85b56d007adef51e5158f5a6dca92794f/dsntnn/__init__.py#L233-L262

    # mu = E[X]
    values = [normalized_linspace(d, dtype=heatmaps.dtype, device=heatmaps.device)  for d in heatmaps.size()[2:]]
    mu = linear_expectation(heatmaps, values)

    # var = E[(X - mu)^2]
    values = [(a - b.squeeze(0)) ** 2 for a, b in zip(values, mu.split(1, -1))]
    var = linear_expectation(heatmaps, values)

    heatmap_size = torch.tensor(list(heatmaps.size()[2:]), dtype=var.dtype, device=var.device)

    return var * (heatmap_size / 2) ** 2