import json
import torch
from common.metrics import evaluate, evaluate_menpo
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def get_cumulative(errors):
    fraction_smaller = []
    sorted_errors = np.sort(errors)
    n = len(errors)

    for error in sorted_errors:
        fraction_smaller.append(len([e for e in sorted_errors if e <= error]) / n)

    return sorted_errors, fraction_smaller


# this is extremely hacky and avoids that we have to change the code that exports the predictions to also export the unchanged GTs in case of a 49 LM model
# menpo error computation needs it
with h5py.File("/home/simon/Desktop/InterACT/Masterarbeit/Code/menpo_1.05.h5") as f:
    d = f["menpo"]
    menpo_gt = torch.tensor(np.array(d["coords"]), dtype=torch.float)



def get_errors(source, split):
    with open(source) as f:
        data = json.load(f)
        preds = []
        gts = []
        errors49 = []
        errors68 = []

        for i, sample in enumerate(data[split]["results"]):
            pred = []
            gt = []

            for landmark in sample["coord_and_variance"]:
                pred.append([landmark["pred_x"], landmark["pred_y"]])
                gt.append([landmark["gt_x"], landmark["gt_y"]])
                vx.append(landmark["var_x"])
                vy.append(landmark["var_y"])
            preds.append(pred)
            gts.append(gt)

            if split == "menpo":

                outline_error, no_outline_error = evaluate_menpo(torch.tensor([pred]), menpo_gt[i:i+1]) # we need 68 LM to compute the error
                errors49.append(no_outline_error / 100)
                errors68.append(outline_error / 100)
            else:
                error = evaluate(torch.tensor([pred]), torch.tensor([gt]))
                errors49.append(error["without_outline"] / 100)
                errors68.append(error["with_outline"] / 100)


        print("NO OUTLINE", np.median(errors49))
        print("OUTLINE", np.median(errors68))
    return errors49, errors68


def plot_multiple(models, split, n_lm):
    plt.grid()
    plt.ylim(0,1)
    plt.yticks(np.arange(0,1.1,0.2))
    plt.ylabel("proportion of images")
    plt.xlabel("normalized RMSE")
    plt.axhline(0.5, linestyle="--", c="#333333", linewidth=1)
    plt.xlim(0.005, 0.04)

    for cat, results in models.items():
        errors = get_errors(results, split)[int(n_lm == 68)]
        if any(e > 100 for e in errors):
            continue # model has not been trained for this
        plt.step(*get_cumulative(errors), label=cat)

    plt.legend()
    plt.show()

basedir = "/tmp/jsons"

model_predictions = {
    "A1": os.path.join(basedir, "e49.json"),
    "A2": os.path.join(basedir, "e68.json"),
    "A3": os.path.join(basedir, "h49.json"),
    "A4": os.path.join(basedir, "h68.json"),
    "B1": os.path.join(basedir, "e.json"),
    "B2": os.path.join(basedir, "h.json"),
    "C1": os.path.join(basedir, "49.json"),
    "C2": os.path.join(basedir, "68.json"),
    "D": os.path.join(basedir, "all.json")
}

#plot_multiple(model_predictions, "menpo", 49)
plot_multiple(model_predictions, "hard", 49)
#plot_multiple(model_predictions, "menpo", 49)
#plot_multiple(model_predictions, "menpo", 68)