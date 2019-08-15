import torch
import scipy.io as sio
from common.metrics import evaluate
import json
import argparse
import os
from datasets.facial_landmarks import FaceLandmarksEasyTestData, FaceLandmarksHardTestData
import h5py
import numpy as np
import cv2
from tools.visualize_predictions import draw_landmarks

parser = argparse.ArgumentParser()
parser.add_argument("pred_src", type=str, help="path to predictions from merget (.mat)")
parser.add_argument("gt_mat_src", type=str, help="path to 300W_test.mat (merget GT)")
parser.add_argument("filenames", type=str, help="path to list of filenames Merget/network/benchmarks/#123/test/img.txt")
parser.add_argument("dataset", type=str, help="Path to 300W .h5 file")
parser.add_argument("target", type=str, help="where to write .json with gt, pred, filename per image and prediction images")
args = parser.parse_args()

pred_src = args.pred_src
gt_mat_src = args.gt_mat_src
filenames = args.filenames
target = args.target

with open(filenames) as f:
    fn_list = [x.split("\t")[0] for x in f.read().splitlines()]

# iBug: 1-135       [0:135]
# LFPW: 136-359     [135:359]
# HELEN: 360-689    [359:689]

ibug_fn = fn_list[0:135]
lfpw_fn = fn_list[135:359]
helen_fn = fn_list[359:689]
helen_lfpw_fn = helen_fn + lfpw_fn

preds = sio.loadmat(pred_src)
xs = preds["xfinal"].transpose(1,0)
ys = preds["yfinal"].transpose(1,0)

coordpreds = []
for i in range(len(xs)):
    tmp = []
    for k in range(len(xs[i])):
        tmp.append((xs[i][k].item(), ys[i][k].item()))
    coordpreds.append(tmp)

ibug_pred = coordpreds[0:135]
lfpw_pred = coordpreds[135:359]
helen_pred = coordpreds[359:689]
helen_lfpw_pred = helen_pred + lfpw_pred

gt_data = sio.loadmat(gt_mat_src)["W300_gt"].transpose(2,1,0).tolist()
ibug_gt = gt_data[0:135]
lfpw_gt = gt_data[135:359]
helen_gt = gt_data[359:689]
helen_lfpw_gt = helen_gt + lfpw_gt

res_ibug = evaluate(torch.tensor(ibug_pred).float(), torch.tensor(ibug_gt).float())
print("ibug", res_ibug)

res_lfpw = evaluate(torch.tensor(lfpw_pred).float(), torch.tensor(lfpw_gt).float())
print("lfpw", res_lfpw)

res_helen = evaluate(torch.tensor(helen_pred).float(), torch.tensor(helen_gt).float())
print("helen", res_helen)

res_helen_lfpw = evaluate(torch.tensor(helen_lfpw_pred).float(), torch.tensor(helen_lfpw_gt).float())
print("helen+lfpw", res_helen_lfpw)

ibug_out = []
for i in range(len(ibug_fn)):
    ibug_out.append({
        "filename": ibug_fn[i],
        "gt": ibug_gt[i],
        "pred": ibug_pred[i]
    })

lfpw_out = []
for i in range(len(lfpw_fn)):
    lfpw_out.append({
        "filename": lfpw_fn[i],
        "gt": lfpw_gt[i],
        "pred": lfpw_pred[i]
    })

helen_out = []
for i in range(len(helen_fn)):
    helen_out.append({
        "filename": helen_fn[i],
        "gt": helen_gt[i],
        "pred": helen_pred[i]
    })

json.dump({
    "ibug": ibug_out,
    "lfpw": lfpw_out,
    "helen": helen_out
}, open(os.path.join(target, "result.json"), "w"))


# Failed attempt to plot Merget predictions:
"""
with h5py.File(args.dataset) as f:
    easy = FaceLandmarksEasyTestData(f)

for i,x in enumerate(easy):
    img = x["original_image"]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lms = x["landmarks"]
    #lms[:,1] *= -1
    scaled_lms = (lms+1)*(96//2)
    print(scaled_lms.shape)

    bi = 0
    b = 1000
    for j in range(len(helen_lfpw_gt)):
        e = np.sum(np.abs(helen_lfpw_gt[j]-scaled_lms))
        if e < b:
            b = e
            bi = j
    print(b, bi)

    gt = helen_lfpw_gt[i]

    img2 = draw_landmarks(img, gt, scale=False, color=[255,0,0])
    #img2 = draw_landmarks(img2, pred, scale=False, color=[0,0,255])
    cv2.imshow("t", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #print(helen_lfpw_fn[i], x["filename"])
"""