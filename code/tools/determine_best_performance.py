import argparse
import torch
from pdm.main import ModelTrainer
from pdm.pdmcore import PDM
import json
from common.util import make_deterministic
from common.metrics import evaluate
import numpy as np
from collections import Counter

PDM.ALPHA_HG_SOURCE = "../" + PDM.ALPHA_HG_SOURCE


def run(pdm, hg_results, gpu):
    location = 'cpu' if gpu is None else "cuda:%d" % gpu

    data = torch.load(pdm, map_location='cpu')
    state_dict = data['state_dict']
    config = data['config']

    make_deterministic(config['random_seed'])

    net = ModelTrainer.create_net(config)
    net.model.load_state_dict(state_dict)
    net.model.eval()
    net.to(location)

    net.bs *= 256

    hg_out = json.load(open(hg_results, "r"))
    #avg_dist = torch.tensor(hg_out["train"]["average_lm_distances"], device=location)
    easy = [x["coord_and_conf"] for x in hg_out["easy"]["results"]]
    easy_gt = torch.tensor([[[y["gt_x"], y["gt_y"]] for y in x] for x in easy], device=location)
    hard = [x["coord_and_conf"] for x in hg_out["hard"]["results"]]
    hard_gt = torch.tensor([[[y["gt_x"], y["gt_y"]] for y in x] for x in hard], device=location)
    train = [x["coord_and_conf"] for x in hg_out["train"]["results"]]

    #gauss = norm(0.0, stddev)
    #easy_hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"], gauss.pdf(y["dist_x"]), gauss.pdf(y["dist_y"])] for y in x] for x in easy], device=location)
    #hard_hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"], gauss.pdf(y["dist_x"]), gauss.pdf(y["dist_y"])] for y in x] for x in hard], device=location)

    import math
    import random

    #mp = lambda x: (-5494.5 * x + 1.099)**2
    #mp = lambda x: 1/(100000*x**2+1)
    mp = lambda x: min(1, max(0, 1/x - 130))
    mp = lambda x : 1/x

    #print(torch.min(avg_dist), torch.max(avg_dist))
    #exit()

    """
    varx = torch.tensor([[1/y["var_x"] for y in x] for x in easy], device=location)
    vary = torch.tensor([[1/y["var_y"] for y in x] for x in easy], device=location)
    print("easy", torch.min(varx), torch.max(varx))
    print("easy", torch.min(vary), torch.max(vary))
    varx = torch.tensor([[1/y["var_x"] for y in x] for x in hard], device=location)
    vary = torch.tensor([[1/y["var_y"] for y in x] for x in hard], device=location)
    print("hard", torch.min(varx), torch.max(varx))
    print("hard", torch.min(vary), torch.max(vary))
    varx = torch.tensor([[1/y["var_x"] for y in x] for x in train], device=location)
    vary = torch.tensor([[1/y["var_y"] for y in x] for x in train], device=location)
    print("train", torch.min(varx), torch.max(varx))
    print("train", torch.min(vary), torch.max(vary))
    exit()
    """

    #easy_hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"], mp(avg_dist[i][0]), mp(avg_dist[i][1])] for i,y in enumerate(x)] for x in easy], device=location)
    #hard_hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"], mp(avg_dist[i][0]), mp(avg_dist[i][1])] for i,y in enumerate(x)] for x in hard], device=location)

    easy_hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"], mp(y["var_x"]), mp(y["var_y"])] for i, y in enumerate(x)] for x in easy], device=location)
    hard_hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"], mp(y["var_x"]), mp(y["var_y"])] for i, y in enumerate(x)] for x in hard], device=location)

    #print(torch.min(easy_hg_pred[:,:,2:]), torch.max(easy_hg_pred[:,:,2:]))
    #print(torch.min(hard_hg_pred[:, :, 2:]), torch.max(hard_hg_pred[:, :, 2:]))
    #exit()

    sample_losses_hg_easy = [np.mean((easy_hg_pred[i,:,:2].cpu().numpy() - easy_gt[i].cpu().numpy())**2) for i in range(easy_gt.shape[0])]

    # TODO test() takes pred and conf now separately
    zs, nr, *_ = net.test(easy_hg_pred, verbose=True)
    l2d_easy, _ = net.forward(zs, nr)

    sample_losses_pdm_easy = [np.mean((l2d_easy[i].detach().cpu().numpy() - easy_gt[i].detach().cpu().numpy()) ** 2) for i in range(easy_gt.shape[0])]

    easy_best = Counter()
    best_coords_easy = []
    worst_coords_easy = []
    for i in range(easy_gt.shape[0]):
        if sample_losses_pdm_easy[i] <= sample_losses_hg_easy[i]:
            easy_best["pdm"] += 1
            best_coords_easy.append(l2d_easy[i].cpu().detach().numpy().tolist())
            worst_coords_easy.append(easy_hg_pred[i, :, :2].cpu().detach().numpy().tolist())
        else:
            easy_best["hg"] += 1
            best_coords_easy.append(easy_hg_pred[i, :, :2].cpu().detach().numpy().tolist())
            worst_coords_easy.append(l2d_easy[i].cpu().detach().numpy().tolist())

    sample_losses_hg_hard = [np.mean((hard_hg_pred[i, :, :2].cpu().numpy() - hard_gt[i].cpu().numpy()) ** 2) for i in range(hard_gt.shape[0])]

    # TODO test() takes pred and conf now separately
    zs, nr, *_ = net.test(hard_hg_pred, verbose=True)
    l2d_hard, _ = net.forward(zs, nr)

    sample_losses_pdm_hard = [np.mean((l2d_hard[i].detach().cpu().numpy() - hard_gt[i].detach().cpu().numpy()) ** 2) for i in range(hard_gt.shape[0])]

    hard_best = Counter()
    best_coords_hard = []
    worst_coords_hard = []
    for i in range(hard_gt.shape[0]):
        if sample_losses_pdm_hard[i] <= sample_losses_hg_hard[i]:
            hard_best["pdm"] += 1
            best_coords_hard.append(l2d_hard[i].cpu().detach().numpy().tolist())
            worst_coords_hard.append(hard_hg_pred[i, :, :2].cpu().detach().numpy().tolist())
        else:
            hard_best["hg"] += 1
            best_coords_hard.append(hard_hg_pred[i, :, :2].cpu().detach().numpy().tolist())
            worst_coords_hard.append(l2d_hard[i].cpu().detach().numpy().tolist())

    hg_easy_eval = evaluate(easy_hg_pred[:, :, :2], easy_gt)
    all_pdm_easy_eval = evaluate(l2d_easy, easy_gt)
    best_pick_easy = evaluate(torch.tensor(best_coords_easy, dtype=torch.float32).cpu(), easy_gt.cpu())
    worst_pick_easy = evaluate(torch.tensor(worst_coords_easy, dtype=torch.float32).cpu(), easy_gt.cpu())
    print("\n---- EASY without outline----")
    print("HG \t\t %0.4f" % hg_easy_eval["without_outline"])
    print("best pick \t %0.4f" % best_pick_easy["without_outline"])
    print("worst pick \t %0.4f" % worst_pick_easy["without_outline"])
    print("all PDM \t %0.4f" % all_pdm_easy_eval["without_outline"])

    print("\n---- EASY with outline----")
    print("HG \t\t %0.4f" % hg_easy_eval["with_outline"])
    print("best pick \t %0.4f" % best_pick_easy["with_outline"])
    print("worst pick \t %0.4f" % worst_pick_easy["with_outline"])
    print("all PDM \t %0.4f" % all_pdm_easy_eval["with_outline"])

    print("easy best", easy_best)

    hg_hard_eval = evaluate(hard_hg_pred[:, :, :2], hard_gt)
    all_pdm_hard_eval = evaluate(l2d_hard, hard_gt)
    best_pick_hard = evaluate(torch.tensor(best_coords_hard, dtype=torch.float32).cpu(), hard_gt.cpu())
    worst_pick_hard = evaluate(torch.tensor(worst_coords_hard, dtype=torch.float32).cpu(), hard_gt.cpu())
    print("\n---- HARD without outline----")
    print("HG \t\t %0.4f" % hg_hard_eval["without_outline"])
    print("best pick \t %0.4f" % best_pick_hard["without_outline"])
    print("worst pick \t %0.4f" % worst_pick_hard["without_outline"])
    print("all PDM \t %0.4f" % all_pdm_hard_eval["without_outline"])

    print("\n---- HARD with outline----")
    print("HG \t\t %0.4f" % hg_hard_eval["with_outline"])
    print("best pick \t %0.4f" % best_pick_hard["with_outline"])
    print("worst pick \t %0.4f" % worst_pick_hard["with_outline"])
    print("all PDM \t %0.4f" % all_pdm_hard_eval["with_outline"])

    print("hard_best", hard_best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hg_results",
                        type=str,
                        help="Inferred coordinates obtained by applying HG")
    parser.add_argument("pdm",
                        type=str,
                        help="Path to PDM")
    parser.add_argument("--gpu",
                        type=int,
                        default=None,
                        help="GPU ID")

    args = vars(parser.parse_args())

    run(args["pdm"], args["hg_results"], args["gpu"])