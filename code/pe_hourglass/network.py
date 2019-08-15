"""
    Author: Simon Hessner (shessner@cs.cmu.edu)

    Classes in this module encapsulate certain behaviour to avoid making the hourglass network code unreadable

    There are multiple regressor variants that transform the heatmaps from the hourglass into coordinates
"""

import torch
import torch.nn as nn
import dsntnn
from pe_hourglass.hourglass import HourglassStack
from collections import OrderedDict
from common.util import heatmap_variance


class Regressor(nn.Module):
    """
    Each Regressor takes inputs in the form of BATCHSIZE x N_COORDS x 32 x 32
    """

    def __init__(self, n_coords):
        super(Regressor, self).__init__()
        self.n_coords = n_coords


class LinearRegressor(Regressor):

    def __init__(self, n_coords):
        super(LinearRegressor, self).__init__(n_coords)

        # When using BatchNorm here (between the last two linear layers), the training is very unstable
        # Often the loss increases rapidly and the training fails. This is why I am only using Dropout here and no BN

        self.layers = nn.Sequential(OrderedDict([
            ('linear1',  nn.Linear(1024, 512)), # 1024 because input is a 32x32 feature map (with N_LM=68 channels)
            ('relu1',    nn.ReLU()),
            ('linear2',  nn.Linear(512, 128)),
            ('relu2',    nn.ReLU()),
            ('linear3',  nn.Linear(128, 2))  # output is a BATCHSIZE x 68 x 2 dimensional vector
        ]))

    def forward(self, heatmaps):
        #print(heatmaps.shape) # BS x N_LM x 32 x 32

        batchsize = heatmaps.shape[0]
        x = heatmaps.view(batchsize, self.n_coords, -1)
        #print(x.shape) # torch.Size([256, 68, 1024]) (256 = batch size, 68 = number of landmarks)

        coords = self.layers(x)

        return coords, heatmaps, heatmaps


class TanhRegressor(LinearRegressor):

    def __init__(self, n_coords):
        super(TanhRegressor, self).__init__(n_coords)
        self.tanh = nn.Tanh()

    def forward(self, heatmap):
        x, heatmap = LinearRegressor.forward(self, heatmap)
        coords = self.tanh(x)
        return coords, heatmap


class DSNTRegressor(Regressor):
    # https://github.com/anibali/dsntnn/blob/master/examples/basic_usage.md

    # TODO change grid of DSNT so that it goes from -1 to 1 (inclusive)?
    # https://github.com/anibali/dsntnn/issues/15

    def __init__(self, n_coords, do_1x1=True):
        super(DSNTRegressor, self).__init__(n_coords=n_coords)

        if do_1x1:
            # TODO do BN?
            self.heatmap_conv = nn.Conv2d(self.n_coords, self.n_coords, kernel_size=1, bias=False)
        else:
            self.heatmap_conv = lambda x: x

    def forward(self, x):
        unnormalized_heatmaps = self.heatmap_conv(x)

        # Softmax converts values in (0,1) range, so no ReLu needed to remove negative values.
        normalized_heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)  # make it a prob distribution
        coords = dsntnn.dsnt(normalized_heatmaps)
        return coords, normalized_heatmaps, unnormalized_heatmaps


class HourglassPipeline(nn.Module):
    """
    Uses a HourglassNet in front of a regression network. The regression network can be changed (e.g. DSNT, linear, tanh)
    """

    def __init__(self, hourglass, regressor, predict_distances=False):
        super(HourglassPipeline, self).__init__()

        self.hourglass = hourglass
        assert isinstance(self.hourglass, HourglassStack), "hourglass must be of type HourglassStack (or subclass)"

        self.regressor = regressor
        assert isinstance(self.regressor, Regressor), "regressor must be of type Regressor (or subclass)"

    def forward(self, x):
        heatmaps = self.hourglass(x)

        if isinstance(heatmaps, list):  # When HG returns intermediate results and not only last HG output
            coords = []
            normalized_heatmaps = []
            unnormalized_heatmaps = []
            var = []

            for _heatmaps in heatmaps:
                _coords, _heatmaps, _unnormalized_heatmaps = self.regressor(_heatmaps)
                coords.append(_coords.clone())
                normalized_heatmaps.append(_heatmaps)
                unnormalized_heatmaps.append(_unnormalized_heatmaps)
                var.append(heatmap_variance(_heatmaps))
        else:  # only use last HG output
            coords, normalized_heatmaps, unnormalized_heatmaps = self.regressor(heatmaps)
            coords = coords.clone()
            var = heatmap_variance(normalized_heatmaps)

        return coords, normalized_heatmaps, var, unnormalized_heatmaps
