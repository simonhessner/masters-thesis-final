import os
import sys
import cv2
import json
import h5py
import torch
from tqdm import tqdm
from common.util import mkdir_if_not_exists
from common.metrics import evaluate, NO_OUTLINE_MASK
from tools.visualize_predictions import draw_landmarks
from datasets.facial_landmarks import FaceLandmarksEasyTestData, FaceLandmarksHardTestData
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def evaluate_predictions_split(split):
    gt = torch.tensor(split["gt"])
    pred = torch.tensor(split["pdm_pred"])
    res = evaluate(pred, gt)
    return res


def evaluate_predictions(predictions):
    return {
        "easy" : evaluate_predictions_split(predictions["easy"]),
        "hard" : evaluate_predictions_split(predictions["hard"])
    }


def check_file(file):
    print("Checking %s" % file)
    with open(file, "r") as f:
        data = json.load(f)
        res_file = data["results"]
        res_now = evaluate_predictions(data["predictions"])

        eps = 10e-6

        passed = True

        if abs(res_now["easy"]["with_outline"] - res_file["pdm_easy68"]) > eps:
            print("easy68 wrong: %0.10f != %0.10f" % (res_now["easy"]["with_outline"], res_file["pdm_easy68"]))
            passed = False
        
        if abs(res_now["hard"]["with_outline"] - res_file["pdm_hard68"]) > eps:
            print("hard68 wrong: %0.10f != %0.10f" % (res_now["hard"]["with_outline"], res_file["pdm_hard68"]))
            passed = False

        if abs(res_now["easy"]["without_outline"] - res_file["pdm_easy49"]) > eps:
            print("easy49 wrong: %0.10f != %0.10f" % (res_now["easy"]["without_outline"], res_file["pdm_easy49"]))
            passed = False
        
        if abs(res_now["hard"]["without_outline"] - res_file["pdm_hard49"]) > eps:
            print("hard49 wrong: %0.10f != %0.10f" % (res_now["hard"]["without_outline"], res_file["pdm_hard49"]))
            passed = False

    return passed


def sanity_check(basedir):
    noencdir = os.path.join(basedir, "no_encoder")
    encdir = os.path.join(basedir, "encoder")

    return all([
        check_file(os.path.join(noencdir, "easy49.json")),
        check_file(os.path.join(noencdir, "hard49.json")),
        check_file(os.path.join(noencdir, "easy68.json")),
        check_file(os.path.join(noencdir, "hard68.json")),
        check_file(os.path.join(encdir, "easy49.json")),
        check_file(os.path.join(encdir, "hard49.json")),
        check_file(os.path.join(encdir, "easy68.json")),
        check_file(os.path.join(encdir, "hard68.json"))
    ])


COLOR_GT = [255,36,255][::-1]
COLOR_HG = [117,206,255][::-1]
COLOR_PDM = [36,255,36][::-1]
COLOR_PDM_INIT = [0,0,225][::-1]


def plot(src, dst, dataset, split):
    mkdir_if_not_exists(dst)

    with open(src, "r") as f:
        data = json.load(f)["predictions"][split]
        pdm_pred = torch.tensor(data["pdm_pred"])
        hg_pred = torch.tensor(data["hg_pred"])

        if "pdm_encoder_pred" in data:
            pdm_enc_pred = torch.tensor(data["pdm_encoder_pred"])
        else:
            pdm_enc_pred = None
        pdm_3d = torch.tensor(data["pdm_3d"])
        gt = torch.tensor(data["gt"])
        n_lm = pdm_pred.shape[1]

        if n_lm == 49:
            gt = gt[:,NO_OUTLINE_MASK,:]

    with h5py.File(dataset) as d:
        if split == "easy":
            dataset = FaceLandmarksEasyTestData(d, n_lm=n_lm)
        else:
            dataset = FaceLandmarksHardTestData(d, n_lm=n_lm)

    for i in tqdm(range(len(dataset))):
        _gt = gt[i]
        img = cv2.cvtColor(dataset[i]["original_image"], cv2.COLOR_RGB2BGR)
        _pdm_pred = pdm_pred[i]
        _hg_pred = hg_pred[i]
        if pdm_enc_pred is not None:
            _pdm_init_pred = pdm_enc_pred[i]
        _pdm_3d = pdm_3d[i]

        cv2.imwrite(os.path.join(dst, "%d_raw.png" % i), img)

        gt_img = draw_landmarks(img, _gt, color=COLOR_GT, size=2)
        cv2.imwrite(os.path.join(dst, "%d_gt.png" % i), gt_img)

        pdm_img = draw_landmarks(img, _pdm_pred, color=COLOR_PDM, size=2)
        cv2.imwrite(os.path.join(dst, "%d_pdm.png" % i), pdm_img)

        pdm_gt_img = draw_landmarks(gt_img, _pdm_pred, color=COLOR_PDM, size=2)
        cv2.imwrite(os.path.join(dst, "%d_pdm_gt.png" % i), pdm_gt_img)

        hg_img = draw_landmarks(img, _hg_pred, color=COLOR_HG, size=2)
        cv2.imwrite(os.path.join(dst, "%d_hg.png" % i), hg_img)

        hg_gt_img = draw_landmarks(gt_img, _hg_pred, color=COLOR_HG, size=2)
        cv2.imwrite(os.path.join(dst, "%d_hg_gt.png" % i), hg_gt_img)

        hg_pdm_img = draw_landmarks(hg_img, _pdm_pred, color=COLOR_PDM, size=2)
        cv2.imwrite(os.path.join(dst, "%d_hg_pdm.png" % i), hg_pdm_img)

        hg_pdm_gt_img = draw_landmarks(hg_gt_img, _pdm_pred, color=COLOR_PDM, size=2)
        cv2.imwrite(os.path.join(dst, "%d_hg_pdm_gt.png" % i), hg_pdm_gt_img)

        if pdm_enc_pred is not None:
            pdm_init_img = draw_landmarks(img, _pdm_init_pred, color=COLOR_PDM_INIT, size=2)
            cv2.imwrite(os.path.join(dst, "%d_pdm_init.png" % i), pdm_init_img)

            pdm_init_pdm_img = draw_landmarks(pdm_init_img, _pdm_pred, color=COLOR_PDM, size=2)
            cv2.imwrite(os.path.join(dst, "%d_pdm_init_and_pdm.png" % i), pdm_init_pdm_img)

            pdm_init_gt_img = draw_landmarks(gt_img, _pdm_init_pred, color=COLOR_PDM_INIT, size=2)
            cv2.imwrite(os.path.join(dst, "%d_pdm_init_and_gt.png" % i), pdm_init_gt_img)

        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.scatter(_pdm_3d[:, 0], _pdm_3d[:, 1] * -1, _pdm_3d[:, 2])
        ax.view_init(elev=70, azim=-90)
        pyplot.savefig(os.path.join(dst, "%d_3d.png" % i))
        pyplot.close()

        res = {
            "gt": _gt.cpu().detach().numpy().tolist(),
            "pdm_2d": _pdm_pred.cpu().detach().numpy().tolist(),
            "pdm_3d": _pdm_3d.cpu().detach().numpy().tolist(),
            "hg": _hg_pred.cpu().detach().numpy().tolist()
        }

        if pdm_enc_pred is not None:
            res["pdm_init_only"] = _pdm_init_pred.cpu().detach().numpy().tolist()

        json.dump(res, open(os.path.join(dst, "%d_predictions.json" % i), "w"), indent=2)
        #return


def plot_all(srcdir, dataset, dstdir):
    plot(os.path.join(srcdir, "encoder", "easy49.json"), os.path.join(dstdir, "encoder", "easy49"), dataset, "easy")
    plot(os.path.join(srcdir, "encoder", "hard49.json"), os.path.join(dstdir, "encoder", "hard49"), dataset, "hard")
    plot(os.path.join(srcdir, "encoder", "easy68.json"), os.path.join(dstdir, "encoder", "easy68"), dataset, "easy")
    plot(os.path.join(srcdir, "encoder", "hard68.json"), os.path.join(dstdir, "encoder", "hard68"), dataset, "hard")
    plot(os.path.join(srcdir, "no_encoder", "easy49.json"), os.path.join(dstdir, "no_encoder", "easy49"), dataset, "easy")
    plot(os.path.join(srcdir, "no_encoder", "hard49.json"), os.path.join(dstdir, "no_encoder", "hard49"), dataset, "hard")
    plot(os.path.join(srcdir, "no_encoder", "easy68.json"), os.path.join(dstdir, "no_encoder", "easy68"), dataset, "easy")
    plot(os.path.join(srcdir, "no_encoder", "hard68.json"), os.path.join(dstdir, "no_encoder", "hard68"), dataset, "hard")

if __name__ == "__main__":
    bd = sys.argv[1]
    if not sanity_check(bd):
        print("There is something wrong with the predictions, do not plot.")
        exit()

    dataset = sys.argv[2]
    target = sys.argv[3]
    plot_all(bd, dataset, target)