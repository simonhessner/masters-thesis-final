import numpy as np
from torchvision.transforms import functional as VisionF
import random
from PIL import Image
import math
import torch
from common.metrics import NO_OUTLINE_MASK

class ImageTransform(object):
    """
    Author: Simon Hessner

    This class can be used to wrap transforms that should only be applied to the image in a image/label sample.
    For example, if your samples look like this:

    {'image' : ..., 'landmarks' : ...} and you have some transformations that should be applied to only the image,
    then wrap this transformation like this:
    ImageTransform(MyTransformation) and you will get this as a result

    {'image' : MyTransformation(...), 'landmarks' ... (unchanged)}

    Use ImageAndLabelTransform for cases where both should be transformed.

    The idea of these two classes is that different types of transformations can be used in one composed transformation,
    e.g.
    transforms.Compose([
            ImageTransform(transforms.ToPILImage()),
            ImageTransform(jitterTransform),
            ImageAndLabelTransform(RandomHorizontalFlip()),
            ImageTransform(transforms.ToTensor()),
            ImageTransform(normTransform)
        ])
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample

    def __repr__(self):
        return "ImageTransform wrapping: %s" % str(self.transform)


class ImageAndLabelTransform(object):
    """
    Author: Simon Hessner

    This class can be used to wrap transforms that should be applied to both the image and the labels (landmarks).
    For example, if your samples look like this:

    {'image' : ..., 'landmarks' : ...} and you have some transformations that should be applied to image and labels,
    then wrap this transformation like this:
    ImageAndLabelTransform(MyTransformation) and you will get this as a result

    ImageAndLabelTransform({'image' : ..., 'landmarks' ... })

    Use ImageTransform for cases where only the image should be transformed

    The idea of these two classes is that different types of transformations can be used in one composed transformation,
    e.g.
    transforms.Compose([
            ImageTransform(transforms.ToPILImage()),
            ImageTransform(jitterTransform),
            ImageAndLabelTransform(RandomHorizontalFlip()),
            ImageTransform(transforms.ToTensor()),
            ImageTransform(normTransform)
    ])
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        to_transform = {'image' : sample['image'],
                        'landmarks' : sample['landmarks'],
                        'face_angle' : sample['face_angle']} # explicitly do not pass the original image!
        transformed = self.transform(to_transform)
        sample.update(transformed)
        return sample

    def __repr__(self):
        return "ImageAndLabelTransform wrapping: %s" % str(self.transform)


class RandomHorizontalFlip(object):
    """
    Horizontally flip the given PIL Image randomly with a given probability.
    If this transformation should be used in combination with image-only transformations, wrap the transformations using
    ImageTransform and ImageAndLabelTransform and then compose them, e.g:

    transforms.Compose([
            ImageTransform(transforms.ToPILImage()),
            ImageTransform(jitterTransform),
            ImageAndLabelTransform(RandomHorizontalFlip()),
            ImageTransform(transforms.ToTensor()),
            ImageTransform(normTransform)
    ])

    Args:
    p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

        # An image with all landmarks can be found here:
        # https://link.springer.com/content/pdf/10.1007%2Fs11263-018-1134-y.pdf
        idx_dict = {1: 17,
                    2: 16,
                    3: 15,
                    4: 14,
                    5: 13,
                    6: 12,
                    7: 11,
                    8: 10,
                    18: 27,
                    19: 26,
                    20: 25,
                    21: 24,
                    22: 23,
                    32: 36,
                    33: 35,
                    37: 46,  # 37 is left eye, 46 right eye (outer-most point)
                    38: 45,
                    39: 44,
                    40: 43,
                    41: 48,
                    42: 47,
                    49: 55,
                    50: 54,
                    51: 53,
                    60: 56,
                    59: 57,
                    61: 65,
                    62: 64,
                    68: 66}
        inv_dict = {v: k for k, v in idx_dict.items()}

        def map_idx(idx):
            idx = idx + 1
            if idx in idx_dict.keys():  # from left to right
                return idx_dict[idx] - 1
            if idx in inv_dict.keys():  # from right to left
                return inv_dict[idx] - 1
            return idx - 1  # no need to mirror, e.g. 28 - 31 (nose back)

        # self._rot=RandomRotation()
        self._inv_idx = np.array([map_idx(i) for i in range(68)])

    def __call__(self, input):
        """
        Args:
            img (PIL Image) and landmarks: Image and landmarks to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            landmarks = input['landmarks']
            if landmarks.shape[0] == 49:
                # Handle case where 49 landmarks are present and need to be mirrored by inserting them into a
                # 68-row tensor and extracting them after the operation
                tmp = np.zeros((68, landmarks.shape[1]))
                tmp[NO_OUTLINE_MASK] = landmarks
                tmp = tmp[self._inv_idx, :]
                landmarks = tmp[NO_OUTLINE_MASK]
            else:
                landmarks = landmarks[self._inv_idx, :]
            landmarks[:, 0] = -1 * landmarks[:, 0]

            input = {
                'image': VisionF.hflip(input['image']),
                'landmarks': landmarks,
                'face_angle' : -1 * input['face_angle']
            }

        return {
            'image': input['image'],
            'landmarks': input['landmarks'],
            'face_angle' : input['face_angle']
        }

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def rotate_point(origin, point, angle):
    """
    Source: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rotate_image(pil_image, angle_degree, retain_scale):
    origsize = pil_image.size
    # rotate counter clockwise
    rotated_image = VisionF.rotate(pil_image, angle=angle_degree, resample=Image.BILINEAR,
                                   expand=not retain_scale)

    if not retain_scale:
        newsize = rotated_image.size  # will be bigger than original image if not 0, 90, 180, 270 or 360 deg
        ratio = origsize[0] / newsize[0]  # used later to scale the landmarks so that they match the rotated image
        rotated_image = VisionF.resize(rotated_image, origsize)  # go back to the original size
    else:
        ratio = 1.0  # since the scale is retained, landmarks only have to be rotated and not scaled

    return rotated_image, ratio


class NormalizeRotation(object):
    def __init__(self, retain_scale=False):
        """
        # TODO doc
        """
        self.retain_scale = retain_scale

    def __call__(self, input):
        angle_radians = input["face_angle"]
        angle_degree = math.degrees(angle_radians)
        image = input["image"]
        landmarks = input["landmarks"]

        rotated_image, ratio = rotate_image(image, angle_degree, retain_scale=self.retain_scale)
        rotated_landmarks = [rotate_point(origin=(0,0), point=p, angle=angle_radians) for p in landmarks]

        #print(angle_degree, angle_radians, ratio)

        return {
            "image" : rotated_image,
            "landmarks" : torch.tensor(rotated_landmarks),
        }


class RandomRotation(object):
    def __init__(self, min_angle, max_angle, origin=(0,0), retain_scale=False, rotate_landmarks="same"):
        """
        Randomly rotate images and (optionally) also landmarks in order to do data augmentation.
        The angle is chosen uniformly from the given range

        :param min_angle: Lower bound for random angle (degree)
        :param max_angle: Upper bound for random angle (degree)
        :param origin: center of rotation, (0,0) by default
        :param retain_scale: When using retain_scale=True, the image is just rotated without rescaling the content,
            leading to cut-off edges. retain_scale=False preserves the whole image, but will shrink it depending on
            the specified angle.
            The image size is not changed in either case.
        :param rotate_landmarks: When False, only the image is rotated. When "neutral", the landmarks are rotated
            so that the eyes are on one line. If "same", the landmarks are rotated exactly like the image.
        """
        assert min_angle <= max_angle, "min_angle <= max_angle not fulfilled"
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.origin = origin
        self.retain_scale = retain_scale
        assert rotate_landmarks in [False, "same", "neutral"]
        self.rotate_landmarks = rotate_landmarks

    def get_angle(self):
        return random.uniform(self.min_angle, self.max_angle)

    def __call__(self, input):
        """
        Rotates image and coordinates of landmarks

        :param input: image and landmarks
        :return: Randomly rotated image and its landmark coords
        """

        angle_degree = self.get_angle()
        rotated_image, ratio = rotate_image(input["image"], angle_degree, self.retain_scale)

        # scale them that they fit the image
        rotated_landmarks = [[x[0], x[1]] for x in input['landmarks'] * ratio]

        # Negation is required, probably due to different coordinate systems
        # (the origin (0,0) of images is the top left corner and the positive coordinates are going right and DOWN,
        # while the origin of "normal" coordinate systems is in the center and the coordinates go right and UP)
        # An alternative to negate the angle is to transform the landmark coordinates before the rotation and to
        # transform them back after the rotation. The transformation would simply set (x,y)=(x,-y) before and
        # after the rotation. But I guess just negating the angle is much easier.
        angle_radians = -input['face_angle'] if self.rotate_landmarks == "neutral" else -math.radians(angle_degree)

        # rotate counter clockwise
        rotated_landmarks = [rotate_point(origin=self.origin, point=p, angle=angle_radians) for p in rotated_landmarks]

        ret = {
            'image': rotated_image,
            'rotated_landmarks': torch.tensor(rotated_landmarks),
            'rot_augm_angle_degree': angle_degree,
            'rot_augm_angle_radians' : math.radians(angle_degree)
        }

        if self.rotate_landmarks is not False:
            ret['landmarks'] = ret['rotated_landmarks']

        return ret
