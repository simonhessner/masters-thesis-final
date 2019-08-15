import os
import json
from matplotlib import pyplot
import numpy as np
from common.util import mkdir_if_not_exists
from tqdm import tqdm
import sys

target = sys.argv[2]


data = json.load(open(sys.argv[1], "r"))
gt = data["gt"]
pdm_pred = data["l2d"]
hg_pred = data["hg"]
losses = data["losses"]
epochs = len(data["l2d"])

pyplot.figure(figsize=(8, 8))

for epoch in tqdm(range(epochs)):
    if epoch < 1400: continue
    for sample_index in tqdm(range(len(hg_pred[epoch]))):
        sample_hg = np.array(hg_pred[epoch][sample_index])
        sample_pdm = np.array(pdm_pred[epoch][sample_index])
        sample_gt = np.array(gt[sample_index])

        pyplot.scatter(sample_gt[:, 0], sample_gt[:, 1] * -1, s=3, label="GT", color="orange")
        pyplot.scatter(sample_hg[:, 0], sample_hg[:, 1] * -1, s=3, label="HG", color="green")
        #pyplot.scatter(sample_pdm[:, 0], sample_hg[:, 1] * -1, s=6, label="PDM", color="blue")

        pyplot.xlim([-1.0, 1.0])
        pyplot.ylim([-1.0, 1.0])

        mkdir_if_not_exists(os.path.join(target, "%d" % sample_index))

        pyplot.legend()
        pyplot.figtext(0.5, 0.01, "Epoch %04d" % (epoch,), wrap=True, horizontalalignment='center', fontsize=12)
        pyplot.savefig(os.path.join(target, "%d/frame_%d.png" % (sample_index, epoch)))
        pyplot.clf()
