import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image, to_tensor
from pe_hourglass.hourglass import HourglassStack

# https://arxiv.org/pdf/1506.02025.pdf
# source https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class STN(nn.Module):
    def __init__(self, dropout=0.5, variant="stn_1", init_regression=True):
        super(STN, self).__init__()

        if variant == "pure_stn":
            self.localization = nn.Sequential( # Spatial transformer localization-network
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),

                nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),

                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),
            )

            self.regression_input_size = 32 * 7 * 7

            self.regression = nn.Sequential(
                nn.Linear(self.regression_input_size, 64),
                nn.Dropout(p=dropout),
                nn.ReLU(inplace=False),
                nn.Linear(64, 2), #rotation angle, scale (same for x and y)
            )

            self.regression_idx = -1
        elif variant == "hg_stn":
            self.hg = HourglassStack(nStack=1,
                                   nModules=1,
                                   nFeats=128,
                                   numOutput=32,
                                   stackDepth=4)

            self.localization = nn.Sequential(  # Spatial transformer localization-network
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),

                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),

                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, stride=1),
            )

            self.regression_input_size = 16 * 20 * 20

            self.regression = nn.Sequential(
                nn.Linear(self.regression_input_size, 32),
                nn.Dropout(p=dropout),
                nn.ReLU(inplace=False),
                nn.Linear(32, 2),  # rotation angle, scale (same for x and y)
            )
            self.regression_idx = -1
        else:
            raise Exception("Invalid STN variant %s" % variant)

        identity_angle = math.radians(0) # counter clockwise
        identity_scale = 1

        if init_regression:
            # Initialize the weights/bias with identity transformation (they are doing it like this in the paper)
            self.regression[self.regression_idx].weight.data.zero_()
            self.regression[self.regression_idx].bias.data.copy_(torch.tensor([identity_angle, identity_scale], dtype=torch.float))

    def get_regression_weight(self):
        """
        Just a function that helps debugging the STN

        :return: weights of final regression layer
        """
        return self.regression[self.regression_idx].weight.data

    def get_regression_bias(self):
        """
        Just a function that helps debugging the STN

        :return: Bias of final regression layer
        """
        return self.regression[self.regression_idx].bias.data

    @staticmethod
    def get_rot_scale_trans(affine_parameters):
        # source: https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix/13165
        a, b, tx, c, d, ty = tuple(affine_parameters)

        sx = np.sign(a) * math.sqrt(a ** 2 + b ** 2)
        sy = np.sign(d) * math.sqrt(c ** 2 + d ** 2)
        angle = math.degrees(math.atan2(c,d))
        if sx == 0.0 or sy == 0.0:
            print("Warning: Scale was 0, so the angle will be incorrect")

        return angle, (sx,sy), (tx,ty)

    @staticmethod
    def build_trans_matrix(radians, scale):
        batchsize = radians.shape[0]
        trans_matrix = torch.stack([
            scale * torch.cos(radians), -1 * scale * torch.sin(radians), torch.zeros(radians.shape).to(radians.device),
            scale * torch.sin(radians),      scale * torch.cos(radians), torch.zeros(radians.shape).to(radians.device),
        ]) # has shape 6 x BS
        trans_matrix = trans_matrix.permute((1,0)) # has shape BS x 6
        trans_matrix = trans_matrix.view((batchsize, 2, 3)) # has shape BS x 2 x 3
        return trans_matrix

    @staticmethod
    def build_inverse_trans_matrix(radians, scale):
        return STN.build_trans_matrix(-1.0 * radians, 1.0 / scale)

    # Spatial transformer network forward function
    def stn(self, x):
        if "hg" in self._modules.keys(): # TODO maybe a bit nicer code here?!?!?!?
            smaller_x = self.hg(x)
        else:
            smaller_x = torch.stack([to_tensor(resize(to_pil_image(i.cpu().permute((1,2,0)).numpy()), (32, 32))) for i in x.type(dtype=torch.uint8)]).to(x.device)

        xs = self.localization(smaller_x)
        #print("before view", xs.shape) # torch.Size([128, 68, 4, 4])

        xs = xs.view(-1, self.regression_input_size)
        #print("after view", xs.shape) # torch.Size([128, 1088])

        trans_params = self.regression(xs)

        # some notes:
        # high scale ==> image gets smaller /// low scale ==> image gets bigger
        # scale between 0.75 and 1.25 is reasonable
        # radians = pi    => degrees = 180
        # radians = 2pi   => degrees = 360
        # radians = pi/2  => degrees = 90
        # radians = pi/3  => degrees = 60
        # radians = pi/4  => degrees = 45
        # radians = 0     => degrees = 0
        # an angle between -90 and 90 degree is reasonable => radians between -pi/2 and pi/2

        radians = trans_params[:,0]
        scale = trans_params[:,1]

        trans_matrix = STN.build_trans_matrix(radians=radians, scale=scale)

        grid = F.affine_grid(trans_matrix, x.size()) # grid should have dimensions x.size()
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode="border") # TODO add random noise or mask Conv

        return x, trans_params

    @staticmethod
    def transform_coords(coords, affine_params):
        """
        Transform coordinates in the same way as the input image was transformed in stn()

        Explanation:
        Assume that the face in the original image was rotated by 10°. The STN would then produce an affine
        transformation that rotates by 10°. By sampling using this transformation, the output of the STN is the
        original image without the rotation (can be seen as the original image rotated by -10°). So the network that
        works on the output of the STN (e.g. Hourglass, DenseNet, etc.) and produces some coordinates based on this
        does not take the rotation of the original image into account.
        This is why the output of the hourglass, densenet, etc. has to be transformed in exactly the same way as the
        original image was transformed by the STN.

        This function iplementes equation 1 of the STN paper with the difference that the source (x^s and y^s) and
        target (x^t and y^t) coordinates were swapped. the reason is that the equation is already formulated as
        "doing the inverse" (the input are the target coordinates and the output are the source coordinates). This is
        required for the sampling mechanism, because for each coordinate in the target we want to know where in the
        source to sample from.

        In order to transform the coordinates, we don't want to do the inverse, so we do the inverse of the inverse,
        which is achieved by swapping the target and source coordinates.

        :param coords: coordinates in a BATCHSIZE x N_COORDS x 2 tensor
        :param theta: Matrix describing the affine transformation that was applied to the input image (see
            function stn(), especially the affine_grid and grid_sample part.
        :return: Coordinates transformed in the same way as the input image
        """
        #print(coords.shape) # 128 x 68 x 2

        coords = coords.clone() # The original coordinates should NOT be transformed!

        trans_matrix = STN.build_inverse_trans_matrix(radians=affine_params[:, 0], scale=affine_params[:, 1])

        relevant_matrix = trans_matrix[:,:,:2]

        coordsT = torch.transpose(coords, 1, 2) # row vector to column vector
        transformed_coords = torch.matmul(relevant_matrix, coordsT)
        transformed_coords = torch.transpose(transformed_coords, 1, 2) # column vector to row vector

        return transformed_coords

    def forward(self, x):
        return self.stn(x)