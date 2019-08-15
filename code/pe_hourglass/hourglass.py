# Original HG code (torch): https://github.com/princeton-vl/pose-hg-train
#
# Good summarization of the Hourglass architecture:
# - https://github.com/aleju/papers/blob/master/neural-nets/Stacked_Hourglass_Networks_for_Human_Pose_Estimation.md
# - http://pocv16.eecs.berkeley.edu/camera_readys/hourglass.pdf
#
# This code is inspired by:
# - https://github.com/princeton-vl/pose-hg-train
# - https://github.com/bearpaw/pytorch-pose
# - https://github.com/xingyizhou/pytorch-pose-hg-3d/tree/2D/
#
# The code and documentation was written by Simon Hessner (hessner@cmu.edu)

import torch.nn as nn
from collections import OrderedDict


class Upsample(nn.Module):
    """
    Fixes a deprecated warning when using torch.nn.Upsample
    Source: https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588

    WARNING: This is not deterministic, even when random seeds and deterministic behaviour are enabled!
    When only nearest neighbor interpolation is needed, NearestNeighborUpsampling should be used instead!
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale = scale_factor
        self.align = align_corners

    def forward(self, x):
        return self.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align, scale_factor=self.scale)


class NearestNeighborUpsampling(nn.Module):
    """
    Deterministic implementation of nearest neighbor upsampling.
    Source: https://github.com/pytorch/pytorch/issues/12207
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        f = self.scale_factor
        # --------------------------------------------------------------------   x has shape [128, 64, 16, 16]
        unsqueezed = x[:, :, :, None, :, None]                                       # shape [128, 64, 16, 1, 16, 1]
        expanded = unsqueezed.expand(-1, -1, -1, f, -1, f)                           # shape [128, 64, 16, 2, 16, 2]
        reshaped = expanded.reshape(x.size(0), x.size(1), x.size(2)*f, x.size(3)*f)  # shape [128, 64, 32, 32]
        return reshaped


class Residual(nn.Module):
    """
    This is the basic building block described in the HG paper. It is used at various places in the stacked hourglass
    network.

    It looks like this:
    Main branch: BN -> ReLU -> 1x1 Conv (0.5*n_ch_out) -> BN -> ReLU -> padded 3x3 Conv (0.5*n_ch_out) -> BN -> ReLU -> 1x1 Conv (n_ch_out)
    shortcut branch: Identity if input and output have the same channel count, otherwise a 1x1 Conv that maps
         from the input to the output dimension

    The output of the residual module is the element-wise sum of the two branches.
    TODO it might also be a good idea to concatenate the two branches and then do another 1x1 convolution to downsize

    Most of the time the output and input dimensions in the stacked HG are the same, exept in these locations:
    - HourglassStack preprocessing res_1
    - HourglassStack preprocessing res_3
    """

    def __init__(self, n_ch_in, n_ch_out):
        super().__init__()
        self.n_channels_in = n_ch_in
        self.n_channels_out = n_ch_out

        # TODO add Dropout?
        self.conv_block = nn.Sequential(OrderedDict([
            ('bn_1',   nn.BatchNorm2d(self.n_channels_in)),
            ('relu_1', nn.ReLU(inplace=True)),
            ('conv_1', nn.Conv2d(self.n_channels_in, self.n_channels_out // 2, kernel_size=1)),  # the factor 2 was used in the original paper's code
            ('bn_2',   nn.BatchNorm2d(self.n_channels_out // 2)),
            ('relu_2', nn.ReLU(inplace=True)),
            ('conv_2', nn.Conv2d(self.n_channels_out // 2, self.n_channels_out // 2, kernel_size=3, stride=1, padding=1)),
            ('bn_3',   nn.BatchNorm2d(self.n_channels_out // 2)),
            ('relu',   nn.ReLU(inplace=True)),
            ('conv_3', nn.Conv2d(self.n_channels_out // 2, self.n_channels_out, kernel_size=1))
        ]))

        # TODO do this always?
        if self.n_channels_in != self.n_channels_out:
            # This is needed to ensure that the amount of channels is the same before concatenating
            self.residual_conv = nn.Conv2d(self.n_channels_in, self.n_channels_out, kernel_size=1)

    def forward(self, x):
        conv_block_out = self.conv_block(x)

        if self.n_channels_in != self.n_channels_out:
            residual_out = self.residual_conv(x)
        else:
            # Input channels and output channels already match, so no need to do 1x1 convs
            residual_out = x

        # Perform element-wise addition of the feature maps (this is possible because they have the same amount of
        # channels)
        out = conv_block_out + residual_out
        return out


class ResidualSequence(nn.Module):
    """
    Wraps a sequence of residual modules that all have the same amount of input and output features
    """

    def __init__(self, n_modules, n_features):
        super().__init__()

        res_modules = []
        for i in range(n_modules):
            res_modules.append(Residual(n_features, n_features))
        self.res_modules = nn.Sequential(*res_modules)

    def forward(self, x):
        return self.res_modules(x)


class Hourglass(nn.Module):
    """
    Implementation of one separate HG that will be used in HourglassStack

    The HG is a sequence of downsampling followed by a sequence of upsampling, combined with "skip-connections"
    between feature maps of the same size. This architecture allows the network to analyze the image at different
    scales and to combine the features of different scales in order to get a global understanding of the image.

    This class implements the HG in a recursive manner. On each recursion level it looks like this:

    input -------- Pool -------- ResSeq -------- HG/ResSeq -------- ResSeq -------- Upsample -------- + --- ResSeq ----- output
      |      (downsize) (pre_bottleneck)       (bottleneck)    (post_bottleneck)    (upsize)          |
      --------------------------------------------- ResSeq --------------------------------------------

    The bottleneck itself is a recursive HG unless the last depth level was reached. In this case it is just another
    sequence of n_res_modules times a residual module with n_features as input and output.

    The branch from the input that goes down works on the input in the original size while the main branch first
    downsizes the image. This way there is information on two scales in each recursion level of the HG. And since
    they are nested all the information will be combined from all scales.

    The final ResSeq is only applied when the HG is not nested in another one. In other words, it is only done at the
    highest level / resolution
    """
    def __init__(self, depth, n_res_mods, n_features, is_nested=False):
        super().__init__()

        self.depth = depth
        self.n_res_mods = n_res_mods
        self.n_features = n_features
        self.is_nested = is_nested

        # This branch works on the original resolution
        self.skip_branch = ResidualSequence(self.n_res_mods, self.n_features)

        # The main branch works on 0.5 times the original resolution
        self.downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pre_bottleneck = ResidualSequence(self.n_res_mods, self.n_features)
        # If the last level has not yet been reached, a nested HG is added as the bottleneck for the main branch
        if self.depth > 1:
            self.bottleneck = Hourglass(depth-1, self.n_res_mods, self.n_features, is_nested=True)
        else:
            self.bottleneck = ResidualSequence(self.n_res_mods, self.n_features)
        self.post_bottleneck = ResidualSequence(self.n_res_mods, self.n_features)
        # TODO try other interpolation methods as well?
        self.upsampler = NearestNeighborUpsampling(scale_factor=2)  # deterministic

        if not self.is_nested:
            # TODO original paper has 2 1x1 convs here, try this instead? Or concatenate the features instead of
            # summing them and then do the 1x1 convs?
            self.final_residual = ResidualSequence(self.n_res_mods, self.n_features)

    def forward(self, x):
        skip_out = self.skip_branch(x)
        main_branch_downsampled = self.downsampler(x)
        main_branch_out = self.pre_bottleneck(main_branch_downsampled)
        main_branch_out = self.bottleneck(main_branch_out)
        main_branch_out = self.post_bottleneck(main_branch_out)
        main_branch_out = self.upsampler(main_branch_out)

        # TODO instead try concatenation and 1x1 convolutions?
        combined = skip_out + main_branch_out
        if self.is_nested:
            out = combined
        else:
            # We are at the highest level (highest image resolution)
            out = self.final_residual(combined)
        return out


class HourglassStack(nn.Module):
    """
    This class implements a full stacked hourglass as described in the paper
    "Alejandro Newell, Kaiyu Yang, and Jia Deng, Stacked Hourglass Networks for Human Pose Estimation, arXiv:1603.06937, 2016."

    The first step is a sequence of a 7x7 convolution, a residual block, a maxpooling layer and two more residual blocks
    The output x of this sequence is fed into the first hourglass network, followed by a 1x1 convolution.
    From here there are two small branches:
    1. produces n_output heatmaps. Can be used to apply an intermediate loss or (in case of the last HG in the stack)
       as the final output. In case of a intermediate HG the heatmaps are mapped back into the feature space by applying
       another 1x1 convolution
    2. The other branch also applies a 1x1 convolution from the feature space into the feature space

    The output of the two branches is added element-wise to the input x and will be used as new input x for the next HG if there is one
    Adding x allows the gradients to flow directly between HGs without having to go through all layers.
    """
    def __init__(self, n_hgs, n_res_modules, n_features, n_output, hg_depth, return_all=False):
        super().__init__()
        self.n_hgs = n_hgs
        self.n_res_modules = n_res_modules
        self.n_features = n_features
        self.n_output = n_output
        self.hg_depth = hg_depth # Orig HG paper fixes this to 4
        self.return_all = return_all  # True => return output of all stacked HGs in a list, otherwise only last

        # TODO investigate effects of smaller initial convolutions (instead of 7x7)
        self.preprocessor = nn.Sequential(OrderedDict([
            ('conv_7x7', nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3)),  # TODO make 64 a variable?
            ('bn_1', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('res_1', Residual(64, 128)),  # TODO make 128 a variable?
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('res_2', Residual(128, 128)),
            ('res_3', Residual(128, self.n_features))
        ]))

        hgs = []
        pre_branch = []
        feat2hm = []
        hm2feat = []
        feat2feat = []
        for i in range(self.n_hgs):
            hgs.append(Hourglass(depth=self.hg_depth, n_res_mods=self.n_res_modules, n_features=self.n_features))
            pre_branch.append(nn.Sequential(
                nn.Conv2d(self.n_features, self.n_features, kernel_size=1, stride=1),
                nn.BatchNorm2d(self.n_features),
                nn.ReLU(inplace=True)
            ))

            feat2hm.append(nn.Conv2d(self.n_features, self.n_output, kernel_size=1, stride=1))

            if i < self.n_hgs - 1:
                feat2feat.append(nn.Conv2d(self.n_features, self.n_features, kernel_size=1, stride=1))
                hm2feat.append(nn.Conv2d(self.n_output, self.n_features, kernel_size=1, stride=1))

        self.hourglasses = nn.ModuleList(hgs)
        self.pre_branch = nn.ModuleList(pre_branch)
        self.feat2hm = nn.ModuleList(feat2hm)
        self.hm2feat = nn.ModuleList(hm2feat)
        self.feat2feat = nn.ModuleList(feat2feat)

    def forward(self, x):
        if self.return_all:
            output = []
        else:
            output = None

        #print(1, x.shape) # N_SAMPLES x 3 x 128 x 128
        x = self.preprocessor(x)
        #print(2, x.shape) # N_SAMPLES x N_FEATURES x 32 x 32

        for i in range(self.n_hgs):
            hg_out = self.hourglasses[i](x)
            branch_root = self.pre_branch[i](hg_out)
            tmp_heatmaps = self.feat2hm[i](branch_root)

            if self.return_all:
                output.append(tmp_heatmaps)
            else:
                output = tmp_heatmaps

            if i < self.n_hgs - 1:
                feat_main = self.feat2feat[i](branch_root)
                feat_branch = self.hm2feat[i](tmp_heatmaps)
                # TODO try concatenation and 1x1 convolution to downsize again?
                x = x + feat_main + feat_branch

        return output  # N_SAMPLES x N_LM x 32 x 32
