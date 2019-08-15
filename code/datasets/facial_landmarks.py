from torchvision import transforms
from torch.utils.data import Dataset
from common.transforms import ImageTransform
import numpy as np
from common.metrics import NO_OUTLINE_MASK
import math

"""
Explanation by Yao Chong (25.1.2019)

Hi Simon,

The train and test set in ceclm_68_split are made by combining the following datasets:
train: helen/trainset, lfpw/trainset, and multipie
test: helen/testset, lfpw/testset, and ibug

The ibug part of the testset is designated as the "hard" test set as the dataset is known to be generally more 
challenging.

The exact indices of the test set are as follows (all inclusive ranges):
helen/testset: [0:329]
lfpw/testset: [330:553]
ibug: [554:688]

As for the overlap between the train and test splits, it seems like the overlap is occurring in the LFPW portion of the 
train and test sets. I don't think I have the exact code I used to create the original datasets, but I think they are 
mostly similar to /media/bighdd4/YaoChong/patch_expert_data/build_data.py
and /media/bighdd4/YaoChong/patch_expert_data/combine_data.py.

Some of the original images and data points are in /media/bighdd4/YaoChong/test_data.


::::::::::TRAIN DATA::::
facial_landmarks_from_holmes_ceclm_68_split.h5 uses LFPW, Helen and multipie in the training set

Helen: [0:1999]
LFPW: [2000:2810]
MultiPIE: [2811:7393]

Note that the original 300-W dataset uses AFW instead of multipie!

hourglass_samples_with_afw_without_multipie.h5 replaces multipie by AFW
"""


class FaceLandmarksDataset(Dataset):
    # These values were computed over the training set and are used to normalize the images before feeding them
    # into the CNN
    # should use these constants instead of calculating them every time again (takes some time)
    # these values are based on the file 300w_1.05.h5
    TRAIN_MEAN = tuple(np.array([142.3320535468328, 106.871614235632, 91.44414660830661]) / 255.0)
    TRAIN_STD = tuple(np.array([67.87115948347929, 59.656749515702906, 58.695009731721555]) / 255.0)

    def __init__(self, source, idxl=None, idxr=None, transform=None, n_lm=68):
        """
        Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

        :param source: h5 data source, must have keys images, coords, filenames
        :param idxl: Left bound for indices in data source (INCLUSIVE). 0 if None
        :param idxr: Right bound for indices in data source (EXCLUSIVE) len(f) if None
        :param transform: Transformations for the image (and labels). Should be
                          common.util.ImageTransform or common.util.ImageAndLabelTransform or a Compose of both
        """

        assert n_lm in [49, 68], "49 or 68 LMs supported"
        if n_lm == 68:
            self.lm_extractor = lambda x: x
        else:
            self.lm_extractor = lambda x: x[NO_OUTLINE_MASK]
        self.n_lm = n_lm

        # for example in /media/bighdd5/YaoChong/patch_expert_data/menpo_68.h5 the keys are not like in the other files
        imgs = source["images"] if "images" in source else source["cropped_imgs"]
        coords = source["coords"] if "coords" in source else source["norm_pts"]

        # handle None
        idxl = idxl or 0
        idxr = idxr or len(imgs)

        self.images = imgs[idxl:idxr]
        self.coords = coords[idxl:idxr]
        self.filenames = source['filenames'][idxl:idxr]
        self.angles = [None]*len(self.images)

        if transform is None:
            transform = ImageTransform(transforms.ToTensor())

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def compute_face_angle(self, landmark):
        left_eye = landmark[36]
        right_eye = landmark[45]
        dx, dy = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])

        if dx == 0.0:
            angle_radians = 0.0
        else:
            angle_radians = math.tan(dy / dx)

        return angle_radians

    def __getitem__(self, idx):
        image = self.images[idx]
        landmark = self.lm_extractor(self.coords[idx])
        filename = self.filenames[idx].decode('utf-8')

        if self.angles[idx] is None:
            self.angles[idx] = self.compute_face_angle(landmark)
        face_angle = self.angles[idx]

        sample = {'image': image,
                  'original_image': image.copy(),
                  'landmarks': landmark,   # could be changed by a transformation (e.g. rotation)
                  'full_landmarks': self.coords[idx], # in case n_lm is 49 this also contains outline   # TODO this is untested
                  #'n_lm': self.n_lm,
                  'face_angle': face_angle,
                  'original_landmarks': landmark,  # will not be changed by transformations
                  'index': idx,
                  'filename': filename}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def undo_normalization(images, mean=TRAIN_MEAN, std=TRAIN_STD):
        """

        :param images: one or more RGB images with values centered around 0. Allowed shapes:
            - N x W x H x 3 (N images)
            - W x H x 3 (1 image)
        :param mean: mean value that was subtracted (will be added now)
        :param std: std value that was divided (will be multiplied now)
        :return: RGB images with values between 0 and 255
        """
        dims = len(images.shape)
        assert dims in [3,4], "shape must be either N x W x H x 3 or W x H x 3"

        if dims == 3:
            images = np.expand_dims(images, 0)
            images = FaceLandmarksDataset.undo_normalization(images, mean, std)
            images = images[0]
            return images
        else:
            images = images.copy().astype(np.float32)
            images[:,:,:] *= std
            images[:,:,:] += mean
            images = (images*255).astype(np.uint8)
            images = np.clip(images, 0, 255)
            return images


    def get_mean_and_std(self):
        """
        Get channel-wise mean and std values of dataset

        :return: mean (0,1), std (0,1), mean (0,255), std (0, 255)
        """
        n = 0
        mean = 0
        std = 0
        for x in self:
            n += 1
            data = x["image"].view(3, -1)
            mean += data.mean(1)
            std += data.std(1)
        mean /= n
        std /= n
        return mean, std, mean*255, std*255


class Menpo(FaceLandmarksDataset):
    split = 'menpo'

    def __init__(self, source, transform=None, idxl=None, idxr=None, n_lm=68):
        super(Menpo, self).__init__(source["menpo"], transform=transform, n_lm=n_lm, idxl=idxl, idxr=idxr)


class FaceLandmarksTrainingData(FaceLandmarksDataset):
    split = 'train'

    def __init__(self, source, transform=None, idxl=None, idxr=None, n_lm=68):
        """
        Yao Chong: The training set consists of helen/trainset, lfpw/trainset, and multipie

        :param source: h5 data source, must have key 'train' with subkeys images, coords, filenames
        :param transform: Transformations for the image (and labels). Should be
                          common.util.ImageTransform or common.util.ImageAndLabelTransform or a Compose of both
        """
        super(FaceLandmarksTrainingData, self).__init__(source['train'], transform=transform, n_lm=n_lm, idxl=idxl, idxr=idxr)


class FaceLandmarksAllTestData(FaceLandmarksDataset):
    split = 'test_all'

    def __init__(self, source, transform=None, n_lm=68):
        """
        Yao Chong:
        The exact indices of the test set are as follows (all inclusive ranges):
        helen/testset: [0:329]
        lfpw/testset: [330:553]
        ibug: [554:688]

        :param source: h5 data source, must have key 'test' with subkeys images, coords, filenames
        :param transform: Transformations for the image (and labels). Should be
                          common.util.ImageTransform or common.util.ImageAndLabelTransform or a Compose of both
        """
        super(FaceLandmarksAllTestData, self).__init__(source['test'], transform=transform, n_lm=n_lm)


class FaceLandmarksEasyTestData(FaceLandmarksDataset):
    split = 'test_easy'

    def __init__(self, source, transform=None, n_lm=68):
        """
        Yao Chong: "easy split" of the test set should be indices 0:553 (inclusive)
        helen/testset: [0:329]
        lfpw/testset: [330:553]

        :param source: h5 data source, must have key 'test' with subkeys images, coords, filenames
        :param transform: Transformations for the image (and labels). Should be
                          common.util.ImageTransform or common.util.ImageAndLabelTransform or a Compose of both
        """
        super(FaceLandmarksEasyTestData, self).__init__(source['test'], idxr=554, transform=transform, n_lm=n_lm)


class FaceLandmarksHardTestData(FaceLandmarksDataset):
    split = 'test_hard'

    def __init__(self, source, transform=None, n_lm=68):
        """
        Yao Chong: "hard split" should be 554:688 (inclusive)
        ibug: [554:688]

        :param source: h5 data source, must have key 'test' with subkeys images, coords, filenames
        :param transform: Transformations for the image (and labels). Should be
                          common.util.ImageTransform or common.util.ImageAndLabelTransform or a Compose of both
        """
        super(FaceLandmarksHardTestData, self).__init__(source['test'], idxl=554, transform=transform, n_lm=n_lm)