import os
import h5py
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from end2end.run_all import E2E, load_pdm, load_hg
from datasets.facial_landmarks import FaceLandmarksEasyTestData, FaceLandmarksHardTestData, FaceLandmarksTrainingData
from torchvision import transforms
from common.transforms import ImageTransform
from torch.utils.data.dataloader import DataLoader
import cv2
from tools.visualize_predictions import draw_landmarks, add_description, gallery, add_border
from common.util import mkdir_if_not_exists
import json

COLOR_GT = [255,36,255][::-1]
COLOR_HG = [117,206,255][::-1]
COLOR_PDM = [36,255,36][::-1]


def sample_mse(predictions, gts):
    return [np.mean((predictions[i] - gts[i]) ** 2) for i in range(gts.shape[0])]


def get_improvements(hg_pred, pdm_pred, gt):
    sample_losses_pdm = sample_mse(pdm_pred.detach().cpu().numpy(), gt.detach().cpu().numpy())
    sample_losses_hg = sample_mse(hg_pred.detach().cpu().numpy(), gt.detach().cpu().numpy())
    improvements = [(i, sample_losses_hg[i], sample_losses_pdm[i], sample_losses_hg[i] / sample_losses_pdm[i]) for i in range(len(sample_losses_hg))]
    sorted_improvements = sorted(improvements, key=lambda x: x[3])
    return sorted_improvements


def plot_best_improvements(hg_pred, pdm_pred, gt, images, target, n=10):
    mkdir_if_not_exists(target)
    improvements = get_improvements(hg_pred, pdm_pred, gt)[-n:]
    all_imgs = []
    all_imgs_desc = []
    for rank, (i, hg_err, pdm_err, ratio) in enumerate(improvements[::-1]):
        print(target, i, hg_err, pdm_err, ratio)
        gt_img = draw_landmarks(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR), gt[i], color=COLOR_GT, size=1)
        cv2.imwrite(os.path.join(target, "%d_%d_gt_%0.4f_plain.png" % (rank, i, ratio)), gt_img)
        gt_img_desc = add_description(gt_img, "GT")
        cv2.imwrite(os.path.join(target, "%d_%d_gt_%0.4f_desc.png" % (rank, i, ratio)), gt_img_desc)

        hg_img = draw_landmarks(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR), hg_pred[i], color=COLOR_HG, size=1)
        cv2.imwrite(os.path.join(target, "%d_%d_hg_%0.4f_plain.png" % (rank, i, ratio)), hg_img)
        hg_img_desc = add_description(hg_img, "HG %0.6f" % hg_err)
        cv2.imwrite(os.path.join(target, "%d_%d_hg_%0.4f_desc.png" % (rank, i, ratio)), hg_img_desc)

        pdm_img = draw_landmarks(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR), pdm_pred[i], color=COLOR_PDM, size=1)
        cv2.imwrite(os.path.join(target, "%d_%d_pdm_%0.4f_plain.png" % (rank, i, ratio)), pdm_img)
        pdm_img_desc = add_description(pdm_img, "PDM %0.6f" % pdm_err)
        cv2.imwrite(os.path.join(target, "%d_%d_pdm_%0.4f_desc.png" % (rank, i, ratio)), pdm_img_desc)

        gal_input = [add_border(x, 5) for x in [gt_img, hg_img, pdm_img]]
        cur_all = gallery(np.array(gal_input), 1)
        cv2.imwrite(os.path.join(target, "%d_%d_all_%0.4f.png" % (rank, i, ratio)), cur_all)

        gal_input_desc = [add_border(x, 5) for x in [gt_img_desc, hg_img_desc, pdm_img_desc]]
        cur_all_desc = gallery(np.array(gal_input_desc), 1)
        cv2.imwrite(os.path.join(target, "%d_%d_all_desc_%0.4f.png" % (rank, i, ratio)), cur_all_desc)

        all_imgs.append(cur_all)
        all_imgs_desc.append(add_description(cur_all_desc, " impr. %0.4f" % ratio))

    all_imgs = gallery(np.array(all_imgs), len(all_imgs))
    cv2.imwrite(os.path.join(target, "gallery.png"), all_imgs)

    all_imgs_desc = gallery(np.array(all_imgs_desc), len(all_imgs_desc))
    cv2.imwrite(os.path.join(target, "gallery_desc.png"), all_imgs_desc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hg", type=str, help="Path to pretrained hourglass (.torch)")
    parser.add_argument("pdm", type=str, help="Path to pretrained PDM (.torch)")
    parser.add_argument("data", type=str, help="all_data_valid_w_profile_pts.h5")
    parser.add_argument("target", type=str, help="Where to store")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--n", type=int, default=10, help="N biggest improvements")
    args = parser.parse_args()

    mkdir_if_not_exists(args.target)

    gpu_id = args.gpu
    location = 'cpu' if gpu_id < 0 else "cuda:%d" % gpu_id

    hg, hg_config = load_hg(args.hg, location)
    pdm, pdm_config = load_pdm(args.pdm, location)
    pdm.print_losses = False
    #pdm.test_epochs = 100

    torch.autograd.set_detect_anomaly(True)  # This makes debugging much easier

    if location is not 'cpu':
        torch.cuda.set_device(torch.device(location))

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    pin_memory = location != 'cpu'
    num_workers = 8

    with h5py.File(args.data, 'r') as f:
        easy_d = FaceLandmarksEasyTestData(f, transform=transform)
        hard_d = FaceLandmarksHardTestData(f, transform=transform)

    e2e = E2E(hg, pdm, 64, max(len(easy_d), len(hard_d)))

    easy = list(DataLoader(dataset=easy_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=len(easy_d)))[0]
    hard = list(DataLoader(dataset=hard_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=len(hard_d)))[0]

    easy_gt = easy["landmarks"]
    easy_images = easy["image"]
    hard_gt = hard["landmarks"]
    hard_images = hard["image"]

    with torch.no_grad():
        easy_pdm, easy_hg, pdm_3d_easy, pdm_affparam_easy = e2e(easy_images.to(location))
        hard_pdm, hard_hg, pdm_3d_hard, pdm_affparam_hard = e2e(hard_images.to(location))

    json.dump({
        "3d_easy" : pdm_3d_easy.detach().cpu().numpy().tolist(),
        "affine_params_easy" : pdm_affparam_easy.detach().cpu().numpy().tolist(),
        "3d_hard" : pdm_3d_hard.detach().cpu().numpy().tolist(),
        "affine_params_hard" : pdm_affparam_hard.detach().cpu().numpy().tolist()
    }, open(os.path.join(args.target, "3dcoords.json"), "w"))

    # get unnormalized images in cv2 format
    easy_images_np = easy["original_image"].numpy()
    hard_images_np = hard["original_image"].numpy()

    plot_best_improvements(easy_hg, easy_pdm, easy_gt, easy_images_np, os.path.join(args.target, "easy"), args.n)
    plot_best_improvements(hard_hg, hard_pdm, hard_gt, hard_images_np, os.path.join(args.target, "hard"), args.n)

