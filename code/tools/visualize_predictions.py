import argparse
import h5py
import torch
from datasets.facial_landmarks import FaceLandmarksTrainingData, FaceLandmarksEasyTestData, FaceLandmarksHardTestData
from common.util import scale_coord, mkdir_if_not_exists
from common.metrics import with_outline_error
from common.transforms import ImageTransform, ImageAndLabelTransform, RandomRotation, RandomHorizontalFlip
from tools.visualize_dataset import draw_landmarks
from torchvision import transforms
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
import multiprocessing

from pe_hourglass.main import ModelTrainer


def gallery(images, cols):
    n, height, width, channels = images.shape

    # if the images can not be displayed in cols columns so that each column has the same amount of images,
    # add so many empty white images that this is the case and then continue normal.
    rest = n % cols
    if rest > 0:
        images = np.append(images, np.ones((rest, height, width, channels), dtype=np.uint8)*255, axis=0)
        n += rest

    rows = n // cols

    tmp_cols = []
    for i in range(rows):
        row_images = images[i*cols:(i+1)*cols]
        tmp_cols.append(np.hstack(row_images))

    return np.vstack(tmp_cols)


def add_border(image, borderwidth):
    height, width, channels = image.shape
    new = np.ones((height + 2*borderwidth, width + 2*borderwidth, channels), dtype=np.uint8)*255
    new[borderwidth:height+borderwidth, borderwidth:width+borderwidth] = image
    return new


def add_description(image, description):
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.5
    fontColor = (0, 0, 0)
    lineType = 1

    text_width, text_height = cv2.getTextSize(description, font, fontScale, lineType)[0]

    height, width, channels = image.shape
    new_image = np.ones((height + text_height + 15, width, channels), dtype=np.uint8) * 255
    new_image[:height,:width] = image

    bottomLeftCornerOfText = (0, new_image.shape[0] - 12)
    cv2.putText(new_image, description, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    return new_image


def get_scaled_error(p1, p2, width):
    p1 = np.array(scale_coord(p1, width))
    p2 = np.array(scale_coord(p2, width))
    return np.linalg.norm(p1 - p2)


def visualize_stn(original, transformed, before_stn, after_stn, filenames, target):
    target = os.path.join(target, "stnviz")
    mkdir_if_not_exists(target)

    transformed = transformed.detach().cpu().numpy().transpose(0,2,3,1)
    before_stn = before_stn.detach().cpu().numpy().transpose(0,2,3,1)
    after_stn = after_stn.detach().cpu().numpy().transpose(0,2,3,1)

    for orig, trans, before, after, filename in zip(original, transformed, before_stn, after_stn, filenames):
        origimg = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        before = cv2.cvtColor(FaceLandmarksTrainingData.undo_normalization(before), cv2.COLOR_RGB2BGR)
        after  = cv2.cvtColor(FaceLandmarksTrainingData.undo_normalization(after), cv2.COLOR_RGB2BGR)
        trans  = cv2.cvtColor(FaceLandmarksTrainingData.undo_normalization(trans), cv2.COLOR_RGB2BGR)

        comb = np.hstack((origimg, trans, before, after))
        fn = os.path.join(target, "%s.png" % filename)
        cv2.imwrite(fn, comb)


def visualize_split(net, split, target, location, landmarks_in_heatmaps):
    mkdir_if_not_exists(target)
    evaluated_images = []

    i = 0

    curidx = 0

    with torch.no_grad():
        for batch in tqdm(split):
            real_landmarks = batch['original_landmarks']
            gt_lm = batch['landmarks']
            original_images = batch['original_image']
            transformed_images = batch['image'].to(location)
            predicted_landmarks, heatmaps, var, unnormal_hms = net(transformed_images)
            #transformed_landmarks = net.stn.transform_coords(predicted_landmarks, affine_params)

            predicted_landmarks = predicted_landmarks.cpu()

            unnormal = FaceLandmarksTrainingData.undo_normalization

            original_images = batch['image'].cpu().detach().permute(0,2,3,1).numpy()

            predicted_heatmaps = heatmaps.cpu().detach().numpy()
            outline_errors = with_outline_error(predicted_landmarks.float(), real_landmarks.float())

            real_landmarks = real_landmarks.detach().numpy()
            predicted_landmarks = predicted_landmarks.detach().numpy()

            COLOR_GT = [255,0,127][::-1]
            COLOR_PRED = [0,0,255][::-1]
            N_LM = 68

            for i in range(len(original_images)):
                original_image = cv2.cvtColor(unnormal(original_images[i]), cv2.COLOR_RGB2BGR).astype(dtype=np.uint8)

                dotsize = 1
                face_gt = draw_landmarks(original_image, real_landmarks[i], color=COLOR_GT, size=dotsize)

                face_prediction = draw_landmarks(original_image, predicted_landmarks[i], color=COLOR_PRED, size=dotsize)
                face_both = draw_landmarks(face_gt, predicted_landmarks[i], color=COLOR_PRED, size=dotsize)

                # upscale heatmaps from 32x32 to 128x128
                imgsize = 128
                scaled_hms = np.array([cv2.resize(hm, dsize=(imgsize, imgsize), interpolation=cv2.INTER_CUBIC) for hm in predicted_heatmaps[i]])

                # make highest value in each heatmap 1 and make it a RGB image
                prob_maps = np.array([scaled_hms[j] / np.max(scaled_hms[j]) for j in range(len(scaled_hms))], dtype=np.float)
                prob_maps = np.stack([prob_maps, prob_maps, prob_maps], axis=3)

                # define a blending max that is used to vizualize the values in the heatmap
                blend = np.ones(prob_maps.shape)
                blend[:,:,:,2] = 0  # R
                blend[:,:,:,1] = 255    # G
                blend[:,:,:,0] = 0  # B

                # interpolate original image and blending mask weighted by the heatmap
                heatmaps = (1 - prob_maps) * original_image + prob_maps * blend
                heatmaps = heatmaps.astype(np.uint8)

                # draw GT and pred
                if landmarks_in_heatmaps:
                    dotsize = 2
                    heatmaps = [draw_landmarks(heatmaps[k], [real_landmarks[i][k]], size=dotsize, color=COLOR_GT) for k in range(N_LM)]
                    heatmaps = [draw_landmarks(heatmaps[k], [predicted_landmarks[i][k]], size=dotsize, color=COLOR_PRED) for k in range(N_LM)]

                # label with variance and error
                errors = [get_scaled_error(real_landmarks[i][k], predicted_landmarks[i][k], imgsize) for k in range(N_LM)]
                variances = [var[i][k].mean() for k in range(N_LM)]
                heatmaps = [add_description(heatmaps[k], "%d V%0.1f E%0.2f" % (k + 1, variances[k], errors[k])) for k in range(N_LM)]


                original_image = add_description(original_image, "input img")
                face_gt = add_description(face_gt, "ground truth")
                face_prediction = add_description(face_prediction, "prediction")
                face_both = add_description(face_both, "GT + pred")
                output_images = np.array([original_image, face_gt, face_prediction, face_both])

                to_show = np.append(output_images, heatmaps, axis=0)
                to_show = np.array([add_border(img, 2) for img in to_show])

                allimages = gallery(to_show, cols=12)

                evaluated_images.append((outline_errors[i].data.item(), allimages, curidx))
                curidx += 1


    print("Write files...")
    ranked_images = sorted(evaluated_images, key=lambda x: x[0])

    #order = [(err, idx) for err,_, idx in ranked_images]
    #print(order)

    for rank, (err, img, _) in tqdm(list(enumerate(ranked_images))):
        cv2.imwrite(os.path.join(target, "%d_%f.png" % (rank, err)), img)


def visualize(model, dataset, target, gpu=None, splits=["easy", "hard"], landmarks_in_heatmaps=True):
    location = 'cpu' if gpu is None else  "cuda:%d" % gpu
    if location is not 'cpu':
        # This fixes the problem that pytorch is always allocating memory on GPU 0 even if this is not included
        # in the list of GPUs to use
        torch.cuda.set_device(torch.device(location))

        # cudnn.benchmark improves training speed when input sizes do not change
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        # It selects the best algorithms as the training iterates over the dataset
        cudnn.benchmark = True

    print("Location: ", location)

    data = torch.load(model, map_location=location)
    state_dict = data['state_dict']
    config = data['config']

    num_workers = multiprocessing.cpu_count()
    batch_size = config['batch_size'] if gpu is not None else num_workers
    pin_memory = gpu is not None

    print("Workers: ", num_workers)
    print("Batchsize: ", batch_size)

    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()

    net = net.to(location)

    mkdir_if_not_exists(target)

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        #ImageAndLabelTransform(RandomHorizontalFlip()),
        #ImageAndLabelTransform(RandomRotation(min_angle=-0, max_angle=0, retain_scale=False)),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    with h5py.File(dataset, 'r') as f:
        if "easy" in splits:
            print("Run on easy")
            easy_d = FaceLandmarksEasyTestData(f, transform=transform)
            #print(len(easy_d))
            easy_loader = DataLoader(dataset=easy_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
            visualize_split(net, easy_loader, os.path.join(target, "easy"), location, landmarks_in_heatmaps=landmarks_in_heatmaps)

        if "hard" in splits:
            print("Run on hard")
            hard_d = FaceLandmarksHardTestData(f, transform=transform)
            #print(len(hard_d))
            hard_loader = DataLoader(dataset=hard_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
            visualize_split(net, hard_loader, os.path.join(target, "hard"), location, landmarks_in_heatmaps=landmarks_in_heatmaps)

        if "train" in splits:
            print("Run on train")
            train = FaceLandmarksTrainingData(f, transform=transform)
            #print(len(train))
            train_loader = DataLoader(dataset=train, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
            visualize_split(net, train_loader, os.path.join(target, "train"), location, landmarks_in_heatmaps=landmarks_in_heatmaps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", type=str, help="Path to pytorch model")
    parser.add_argument("dataset", type=str, help="Path to dataste h5 file")
    parser.add_argument("target", type=str, help="Target directory")
    parser.add_argument("--gpu", default=None, type=int, help="Run on GPU ID --gpu instead of CPU")
    parser.add_argument("--splits", default=["easy", "hard"], nargs="+", choices=["easy", "hard", "train"])
    parser.add_argument("--no_landmarks_in_heatmaps", action="store_true")
    args = parser.parse_args()

    visualize(model=args.model, dataset=args.dataset, target=args.target, gpu=args.gpu, splits=args.splits, landmarks_in_heatmaps=not args.no_landmarks_in_heatmaps)