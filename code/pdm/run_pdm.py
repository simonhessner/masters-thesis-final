import json
import torch
import os
from pdm.main import ModelTrainer
from common.metrics import evaluate
from common.util import mkdir_if_not_exists
from pdm.encoder import Encoder
import argparse
from common.util import make_deterministic

from pdm.pdmcore import PDM
PDM.ALPHA_HG_SOURCE_49 = "../" + PDM.ALPHA_HG_SOURCE_49
PDM.ALPHA_HG_SOURCE_68 = "../" + PDM.ALPHA_HG_SOURCE_68


def run_pdm(pdm_path, hg_results, location=torch.device("cpu"), bs=512, encoder=None, history=False):
    data = torch.load(pdm_path, map_location=location)
    state_dict = data['state_dict']
    config = data['config']

    # Altough the same random seed is used as in training, the results will slightly differ
    # The reason is that the metrics are calculated after the training already run for a few epochs, so the
    # random number generator will be in a different state depending on the training before
    make_deterministic(config['random_seed'])

    pdm = ModelTrainer.create_net(config)
    pdm.model.load_state_dict(state_dict)
    pdm = pdm.to(location)

    if encoder is not None:
        enc_data = torch.load(encoder)
        encoder = Encoder(zs_size=enc_data["zs_size"], nr_size=enc_data["nr_size"])
        encoder.load_state_dict(enc_data["state_dict"])
        encoder = encoder.to(location)
    print("Encoder", encoder)

    hg_coords = torch.tensor([[[lm["pred_x"], lm["pred_y"]] for lm in sample["coord_and_conf"]] for sample in hg_results], device=location)
    gt = torch.tensor([[[lm["gt_x"], lm["gt_y"]] for lm in sample["coord_and_conf"]] for sample in hg_results], device=location)
    #variances = torch.tensor([[[lm["var_x"], lm["var_y"]] for lm in sample["coord_and_conf"]] for sample in hg_results], device=location)
    hg_coords_and_conf = torch.tensor([[[lm["pred_x"],
                                         lm["pred_y"],
                                         pdm.variance2confidence(lm["var_x"]),
                                         pdm.variance2confidence(lm["var_y"])] for lm in sample["coord_and_conf"]] for sample in hg_results], device=location)


    if history:
        # TODO test() takes pred and conf now separately
        zs, nr, _, history = pdm.test(hg_coords_and_conf, return_history=True, encoder=encoder, only_encoder=False, bs=bs)
    else:
        # TODO test() takes pred and conf now separately
        zs, nr, *_ = pdm.test(hg_coords_and_conf, return_history=False, encoder=encoder, only_encoder=False, bs=bs)
        history = None
    l2d, _ = pdm.forward(zs, nr)
    l2d = l2d.detach()

    return hg_coords, hg_coords_and_conf, gt, l2d, history


if __name__ == "__main__": # TODO add same options as in gapsearch/main.py
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdm", type=str, help="Path to PDM (.torch)")
    parser.add_argument("hg_results", type=str, help="Path to hourglass output (.json)")
    parser.add_argument("--encoder", type=str, help="Path to encoder (.torch)")
    parser.add_argument("--target", type=str, help="Directory to store results in .json file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 = CPU")
    parser.add_argument("--bs", type=int, default=2048, help="How many samples to process in parallel")

    args = parser.parse_args()

    location = torch.device("cuda:%d" % args.gpu if args.gpu >= 0 else "cpu")

    if args.target is not None:
        mkdir_if_not_exists(args.target)
        needs_history = True
    else:
        needs_history = False

    pdm_path = args.pdm
    hg_results = json.load(open(args.hg_results, "r"))

    hg_coords_easy, hg_coords_and_conf_easy, gt_easy, l2d_easy, history_easy = run_pdm(pdm_path=pdm_path,
                                                                                       hg_results=hg_results["easy"],
                                                                                       location=location,
                                                                                       encoder=args.encoder,
                                                                                       history=needs_history,
                                                                                       bs=args.bs)
    hg_coords_hard, hg_coords_and_conf_hard, gt_hard, l2d_hard, history_hard = run_pdm(pdm_path=pdm_path,
                                                                                       hg_results=hg_results["hard"],
                                                                                       location=location,
                                                                                       encoder=args.encoder,
                                                                                       history=needs_history,
                                                                                       bs=args.bs)

    res_easy_before = evaluate(hg_coords_easy, gt_easy)
    easy_with_outline_before = res_easy_before["with_outline"]
    easy_without_outline_before = res_easy_before["without_outline"]
    res_easy_after = evaluate(l2d_easy, gt_easy)
    easy_with_outline_after = res_easy_after["with_outline"]
    easy_without_outline_after = res_easy_after["without_outline"]

    res_hard_before = evaluate(hg_coords_hard, gt_hard)
    hard_with_outline_before = res_hard_before["with_outline"]
    hard_without_outline_before = res_hard_before["without_outline"]
    res_hard_after = evaluate(l2d_hard, gt_hard)
    hard_with_outline_after = res_hard_after["with_outline"]
    hard_without_outline_after = res_hard_after["without_outline"]

    print("[before PDM] easy without \t", easy_without_outline_before)
    print("[before PDM] easy with \t\t", easy_with_outline_before)
    print("[before PDM] hard without \t", hard_without_outline_before)
    print("[before PDM] hard with \t\t", hard_with_outline_before)

    print("[after PDM]  easy without \t", easy_without_outline_after, "\t", easy_without_outline_after/easy_without_outline_before)
    print("[after PDM]  easy with \t\t", easy_with_outline_after, "\t", easy_with_outline_after/easy_with_outline_before)
    print("[after PDM]  hard without \t", hard_without_outline_after, "\t", hard_without_outline_after/hard_without_outline_before)
    print("[after PDM]  hard with \t\t", hard_with_outline_after, "\t", hard_with_outline_after/hard_with_outline_before)

    if args.target is not None:
        print("\nDump results to", args.target)
        json.dump({
            "easy_l2d": l2d_easy.cpu().numpy().tolist(),
            "easy_before_pdm" : hg_coords_easy.detach().cpu().numpy().tolist(),
            "easy_gt" : gt_easy.detach().cpu().numpy().tolist(),
            "hard_l2d": l2d_hard.cpu().numpy().tolist(),
            "hard_before_pdm": hg_coords_hard.detach().cpu().numpy().tolist(),
            "hard_gt": gt_hard.detach().cpu().numpy().tolist(),
            #"history" : history,
            #"confidence" : easy_coords_and_conf[:,:,2:].detach().cpu().numpy().tolist()
            }, open(os.path.join(args.target, "pdm_results.json"), "w"))
