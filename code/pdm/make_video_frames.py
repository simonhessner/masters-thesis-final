import json
from matplotlib import pyplot
import numpy as np
from common.util import mkdir_if_not_exists
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pdm.run_pdm import run_pdm
import os
import argparse


def make_video(pdm_path, hg_results, target, include=["gt", "hg", "pdm", "connection_gt", "connection_hg", "confidence"], epoch_frequency=1, encoder=None):
    hg_coords, hg_coords_and_conf, gt, l2d, history = run_pdm(pdm_path, hg_results, encoder=encoder, history=True)

    if "pdm" not in include:
        print("Warning: pdm output is not included in video, so each frame will be the same")

    gt = gt.detach().cpu().numpy()
    hg_out = hg_coords.detach().cpu().numpy()

    losses = [mean_squared_error(hg_out[i], history[-1]["l2d"][i]) for i in range(len(hg_out))]
    sortedlosses = list(sorted(losses, reverse=True))[:2]
    interesting_losses_idx = [i for i in range(len(losses)) if losses[i] in sortedlosses]

    pyplot.figure(figsize=(8, 8))

    for sample_index in tqdm(interesting_losses_idx):
        tqdm.write(str(sample_index))
        evolution = np.array([history[epoch]["l2d"][sample_index] for epoch in range(len(history))][::epoch_frequency])
        mkdir_if_not_exists(os.path.join(target, "%d" % sample_index), verbose=False)

        framenr = 0
        for epoch, pred in tqdm(list(enumerate(evolution))):
            epoch *= epoch_frequency
            mse = mean_squared_error(hg_out[sample_index], pred)

            if "hg" in include:
                pyplot.scatter(hg_out[sample_index, :, 0], hg_out[sample_index, :, 1] * -1, s=6, label="HG", color="green")

            if "gt" in include:
                pyplot.scatter(gt[sample_index, :, 0], gt[sample_index, :, 1] * -1, s=6, label="GT", color="orange")

            if "pdm" in include:
                pyplot.scatter(pred[:, 0], pred[:, 1] * -1, s=8, label="PDM", color="blue")

            for i in range(68):
                if "connection_gt" in include:
                    x1, y1 = [gt[sample_index, i, 0], pred[i, 0]], [gt[sample_index, i, 1] * -1, pred[i, 1] * -1]
                    pyplot.plot(x1, y1, marker=None, color='orange', linewidth=1, linestyle='dashed')

                if "connection_hg" in include:
                    x1, y1 = [hg_out[sample_index, i, 0], pred[i, 0]], [hg_out[sample_index, i, 1] * -1, pred[i, 1] * -1]
                    pyplot.plot(x1, y1, marker=None, color='green', linewidth=1, linestyle='dashed')

                if "pdm" in include and "confidence" in include:
                    pyplot.annotate("%0.1f" % sum(hg_coords_and_conf[sample_index,i,2:]), (pred[i, 0], pred[i, 1] * -1), size=6)

            pyplot.xlim([-1.0, 1.0])
            pyplot.ylim([-1.0, 1.0])

            pyplot.legend()
            pyplot.figtext(0.5, 0.01, "Epoch %04d - Loss %0.4f" % (epoch, mse), wrap=True, horizontalalignment='center', fontsize=12)
            pyplot.savefig(os.path.join(target, "%d/frame_%d.png" % (sample_index, framenr)))
            pyplot.clf()
            framenr += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdm", type=str, help="Path to PDM (.torch)")
    parser.add_argument("hourglass_results", type=str, help="Path to Hourglass results (.json)")
    parser.add_argument("target", type=str, help="Where to store the frames as .png")
    parser.add_argument("--splits", nargs="+", type=str, default=["easy", "hard"], choices=["easy", "hard", "train"],
                        help="Which splits to run")
    parser.add_argument("--plot", type=str, nargs="+", default=["gt", "hg", "pdm", "connection", "confidence"],
                        choices=["gt", "hg", "pdm", "connection_gt", "connection_hg", "confidence"], help="What to include in frames")
    parser.add_argument("--epoch_frequency", type=int, default=1, metavar="N", help="Plot only every N epochs")
    parser.add_argument("--encoder", type=str, help="Path to encoder (.torch)")

    args = parser.parse_args()
    # TODO offer option to specify which images to plot

    target_dir = args.target
    mkdir_if_not_exists(target_dir)

    hg_results = json.load(open(args.hourglass_results, "r"))

    for split in set(args.splits):
        target = os.path.join(target_dir, split)
        mkdir_if_not_exists(os.path.join(target_dir, target))
        make_video(pdm_path=args.pdm, hg_results=hg_results[split], target=target, include=list(set(args.plot)), epoch_frequency=args.epoch_frequency, encoder=args.encoder)