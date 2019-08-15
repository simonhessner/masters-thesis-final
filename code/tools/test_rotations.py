import argparse
from datasets.facial_landmarks import FaceLandmarksTrainingData, FaceLandmarksEasyTestData, FaceLandmarksHardTestData
import torch
from torchvision import transforms
from common.transforms import ImageTransform, ImageAndLabelTransform, RandomRotation, NormalizeRotation, RandomHorizontalFlip
import h5py
import cv2
import numpy as np
from tools.visualize_dataset import draw_landmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",
                        type=str,
                        default="/home/simon/Desktop/InterACT/Masterarbeit/Code/facial_landmarks_from_holmes_ceclm_68_split.h5",
                        help="Path to dataste h5 file")
    args = parser.parse_args()

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        #ImageAndLabelTransform(RandomHorizontalFlip()),
        #ImageAndLabelTransform(NormalizeRotation()),
        ImageAndLabelTransform(RandomRotation(min_angle=-30, max_angle=30, retain_scale=False, rotate_landmarks="neutral")),
        ImageTransform(transforms.ToTensor()),
        #ImageTransform(normTransform)
    ])

    with h5py.File(args.dataset, 'r') as f:
        easy_d = FaceLandmarksEasyTestData(f, transform=transform)
        hard_d = FaceLandmarksHardTestData(f, transform=transform)
        train = FaceLandmarksTrainingData(f, transform=transform)

        imgs = []
        for x in easy_d:
            imgs.append((x["angle"],
                         x["original_image"],
                         (255 * x["image"]).type(torch.uint8).permute(1, 2, 0).numpy(),
                         x["original_landmarks"],
                         x["landmarks"]
            ))

        cv2.namedWindow("test_rotations")  # Create a named window
        cv2.moveWindow("test_rotations", 200, 200)

        imgs = sorted(imgs, key=lambda x: abs(x[0]))
        for angle, original_image, corrected_img, origlms, lms in imgs[::-1][:3]:
            withlms = draw_landmarks(corrected_img[:, :, ::-1], lms)
            origwithlms = draw_landmarks(original_image[:, :, ::-1], origlms)
            cv2.imshow("test_rotations", np.concatenate((origwithlms, withlms), axis=1))
            cv2.waitKey()
        cv2.destroyAllWindows()