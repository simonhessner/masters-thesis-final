import h5py
import cv2
from tqdm import  tqdm
import os
import argparse
from common.util import mkdir_if_not_exists, scale_coord
from datasets.facial_landmarks import FaceLandmarksTrainingData, FaceLandmarksEasyTestData, FaceLandmarksHardTestData
from datasets.facial_landmarks import Menpo
from collections import defaultdict


def draw_landmarks(image, landmarks, color=(0, 0, 255), size=2, scale=True):
    image = image.copy() # otherwise the original image would be changed
    width,height,_ = image.shape
    assert width == height, "width (%d) must be the same as height (%d)" % (width, height)

    for coord in landmarks:
        if scale:
            converted_coord = scale_coord(coord, width)
        else:
            converted_coord = tuple(int(x) for x in coord)
        cv2.circle(image, converted_coord, size, color, -1, cv2.LINE_AA)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show landmarks as yellow dots on faces.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     allow_abbrev=False)  # For example, this will not accept --growth where --growth_rate is expected

    parser.add_argument('--source',
                        type=str,
                        default='/media/bighdd7/Holmes/ceclm_68_split.h5',
                        help='h5 file containing training and test split')
    parser.add_argument('--target',
                        type=str,
                        default='/tmp/converted_landmarks',
                        help='Images will be saved here')

    parser.add_argument('--no_landmarks',
                        dest='draw_landmarks',
                        default=True,
                        action='store_false',
                        help='Store original images instead of landmarks')

    parser.add_argument('--is_menpo',
                        dest="is_menpo",
                        default=False,
                        action="store_true",
                        help="Specify this when using menpo instead of 300-W")

    args = parser.parse_args()

    with h5py.File(args.source, 'r') as f:
        if args.is_menpo:
            splits = [Menpo(f)]
        else:
            splits = [FaceLandmarksTrainingData(f), FaceLandmarksEasyTestData(f), FaceLandmarksHardTestData(f)]

        stats = defaultdict(int)

        for data in splits:
            split = data.split
            print('Split: %s' % split)

            directory = os.path.join(args.target, split)
            mkdir_if_not_exists(directory)

            for sample in tqdm(list(data)):
                image = sample['original_image'][:,:,::-1].copy()  # RGB o BGR

                if args.draw_landmarks:
                    image = draw_landmarks(image, sample['landmarks'])

                target_file = os.path.join(directory, sample['filename'])
                cv2.imwrite(target_file, image)

                stats[split] += 1

        print("Dataset stats")
        for split, count in stats.items():
            print("%s : %d images" % (split, count))