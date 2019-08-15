import os
import sys
import cv2
import json
import h5py
import torch
from tqdm import tqdm
from common.util import mkdir_if_not_exists
from common.metrics import NO_OUTLINE_MASK
from tools.visualize_predictions import draw_landmarks
from datasets.facial_landmarks import FaceLandmarksEasyTestData, FaceLandmarksHardTestData

source = sys.argv[1]
dataset = sys.argv[2]
target = sys.argv[3]

COLOR_GT = [255,36,255][::-1]
COLOR_HG = [117,206,255][::-1]
COLOR_ENCODER = [36,255,36][::-1]


def plot(src, dst, dataset, split):
    mkdir_if_not_exists(dst)

    with open(src, "r") as f:
        data = json.load(f)["predictions"][split]
        enc_pred = torch.tensor(data["encoder_pred"])
        hg_pred = torch.tensor(data["hg_pred"])

        gt = torch.tensor(data["gt"])
        n_lm = gt.shape[1]

        #if n_lm == 49:
        #    gt = gt[:,NO_OUTLINE_MASK,:]

    with h5py.File(dataset) as d:
        if split == "easy":
            dataset = FaceLandmarksEasyTestData(d, n_lm=n_lm)
        else:
            dataset = FaceLandmarksHardTestData(d, n_lm=n_lm)

    for i in tqdm(range(len(dataset))):
        _gt = gt[i]
        img = cv2.cvtColor(dataset[i]["original_image"], cv2.COLOR_RGB2BGR)
        _enc_pred = enc_pred[i]
        _hg_pred = hg_pred[i]

        cv2.imwrite(os.path.join(dst, "%d_raw.png" % i), img)

        gt_img = draw_landmarks(img, _gt, color=COLOR_GT, size=2)
        cv2.imwrite(os.path.join(dst, "%d_gt.png" % i), gt_img)

        hg_img = draw_landmarks(img, _hg_pred, color=COLOR_HG, size=2)
        cv2.imwrite(os.path.join(dst, "%d_hg.png" % i), hg_img)

        hg_gt_img = draw_landmarks(gt_img, _hg_pred, color=COLOR_HG, size=2)
        cv2.imwrite(os.path.join(dst, "%d_hg_gt.png" % i), hg_gt_img)

        enc_img = draw_landmarks(img, _enc_pred, color=COLOR_ENCODER, size=2)
        cv2.imwrite(os.path.join(dst, "%d_encoder.png" % i), enc_img)

        enc_gt_img = draw_landmarks(gt_img, _enc_pred, color=COLOR_ENCODER, size=2)
        cv2.imwrite(os.path.join(dst, "%d_enc_gt.png" % i), enc_gt_img)

        enc_hg_img = draw_landmarks(hg_img, _enc_pred, color=COLOR_ENCODER, size=2)
        cv2.imwrite(os.path.join(dst, "%d_enc_hg.png" % i), enc_hg_img)


        res = {
            "gt": _gt.cpu().detach().numpy().tolist(),
            "encoder": _enc_pred.cpu().detach().numpy().tolist(),
            "hg": _hg_pred.cpu().detach().numpy().tolist()
        }

        json.dump(res, open(os.path.join(dst, "%d_predictions.json" % i), "w"), indent=2)


plot(os.path.join(source, "easy49.json"), os.path.join(target, "easy49"), dataset, "easy")
plot(os.path.join(source, "hard49.json"), os.path.join(target, "hard49"), dataset, "hard")
plot(os.path.join(source, "easy68.json"), os.path.join(target, "easy68"), dataset, "easy")
plot(os.path.join(source, "hard68.json"), os.path.join(target, "hard68"), dataset, "hard")