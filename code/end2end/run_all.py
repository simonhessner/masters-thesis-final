import argparse
import torch
from pe_hourglass.main import ModelTrainer as HGModelTrainer
from pdm.main import ModelTrainer as PDMModelTrainer
from datasets.facial_landmarks import FaceLandmarksEasyTestData, FaceLandmarksTrainingData, FaceLandmarksHardTestData
from datasets.facial_landmarks import Menpo
from common.transforms import ImageTransform
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import h5py
from common.metrics import evaluate
import torch.nn as nn
from pdm.pdmcore import PDM
from pdm.encoder import Encoder
from common.util import make_deterministic, move2device
import os
import json
from common.metrics import evaluate_menpo


# TODO move to own file
# TODO enable reading hG results from a cache?
class E2E(nn.Module):
    def __init__(self, hg, pdm:PDM, hg_bs, pdm_bs, encoder=None, verbose=True, var_thresh=None):
        super(E2E, self).__init__()
        self.hg = hg
        self.pdm = pdm
        self.hg_bs = hg_bs
        self.pdm_bs = pdm_bs
        self.encoder = encoder
        self.verbose = verbose
        self.variance_threshold = var_thresh

    def forward(self, input):
        self.hg.eval()

        # -------- HG START -----------
        # Run HG in smaller batches because it needs to fit into memory
        hg_out_chunks = []
        hg_conf_chunks = []
        hg_var_chunks = []
        for chunk in input.split(self.hg_bs):
            _hg_out, _, _hg_var, _ = self.hg(chunk)
            _hg_confidence = self.pdm.variance2confidence(_hg_var)
            hg_out_chunks.append(_hg_out)
            hg_conf_chunks.append(_hg_confidence)
            hg_var_chunks.append(_hg_var)
        hg_prediction = torch.cat(hg_out_chunks)
        hg_confidence = torch.cat(hg_conf_chunks)
        hg_variance = torch.cat(hg_var_chunks)

        # -------- HG END -----------

        # -------- PDM START -----------
        # The PDM is a tiny model, so it can run the whole dataset in one run which speeds up everything a lot!

        if self.verbose: print("var thresh", self.variance_threshold)

        if self.variance_threshold is not None:
            # The PDM will only be applied to some samples. For the others, the end result will be the HG pred.
            # Copy the whole set of HG preds and override the ones where the PDM is applied later.
            pdm_result = hg_prediction.clone()
            pdm_res_3d = torch.ones((pdm_result.shape[0], pdm_result.shape[1], 3), device=pdm_result.device) * -10.0 # -10 signals that the PDM was not used
            pdm_zs = torch.ones((pdm_result.shape[0], self.pdm.zdim), device=pdm_result.device) * -10.0 # -10 signals that the PDM was not used
            pdm_nr = torch.ones((pdm_result.shape[0], 9), device=pdm_result.device) * -10.0 # -10 signals that the PDM was not used
            pdm_encoder_result = hg_prediction.clone()

            var = hg_variance.view(hg_variance.shape[0], -1)
            ignore_pdm = var.max(dim=1)[0] < self.variance_threshold
            apply_pdm = 1-ignore_pdm
            remaining_samples = apply_pdm.sum().data.item()
            if self.verbose: print("PDM applied to", remaining_samples, "of", var.shape[0])
            if remaining_samples == 0:
                return pdm_result, hg_prediction, pdm_res_3d, pdm_zs, pdm_nr, apply_pdm
            pdm_input_data = hg_prediction[apply_pdm]
            pdm_input_conf = hg_confidence[apply_pdm]
        else:
            pdm_input_data = hg_prediction
            pdm_input_conf = hg_confidence

        with torch.enable_grad():
            zs, nr, *_, encoder_zs, encoder_nr = self.pdm.test(pdm_input_data, pdm_input_conf, verbose=self.verbose, bs=self.pdm_bs, encoder=self.encoder)
        pdm_prediction, shape_3d = self.pdm.forward(zs, nr)

        if encoder_zs is not None and encoder_nr is not None:
            pdm_encoder_pred, *_ = self.pdm.forward(encoder_zs, encoder_nr)
        else:
            pdm_encoder_pred = None

        if self.variance_threshold is None:
            apply_pdm = torch.tensor(list(range(pdm_prediction.shape[0])))
            return pdm_prediction, hg_prediction, shape_3d, zs, nr, apply_pdm, pdm_encoder_pred
        else:
            pdm_result[apply_pdm] = pdm_prediction
            pdm_res_3d[apply_pdm] = shape_3d
            pdm_zs[apply_pdm] = zs
            pdm_nr[apply_pdm] = nr
            if pdm_encoder_pred is not None:
                pdm_encoder_result[apply_pdm] = pdm_encoder_pred
            else:
                pdm_encoder_result[:] = -1000.0
            return pdm_result, hg_prediction, pdm_res_3d, pdm_zs, pdm_nr, apply_pdm, pdm_encoder_result

    def eval(self):
        self.hg.eval()
        self.pdm.model.eval()

    def train(self):
        self.hg.train()
        self.pdm.model.train()


def load_hg(path, location):
    data = torch.load(path, map_location=location)
    state_dict = data['state_dict']
    config = data['config']

    hg = HGModelTrainer.create_net(config, verbose=False)
    hg.load_state_dict(state_dict)
    hg.eval()

    hg = hg.to(location)

    return hg, config


def load_pdm(path, location):
    data = torch.load(path, map_location=location)
    state_dict = data['state_dict']
    config = data['config']

    pdm = PDMModelTrainer.create_net(config)
    pdm.model.load_state_dict(state_dict)
    pdm = pdm.to(location)
    return pdm, config


def load_encoder(path, location, n_coords=68):
    if path is not None:
        path = os.path.expanduser(path)
        enc_data = torch.load(path)
        encoder = Encoder(zs_size=enc_data["zs_size"], nr_size=enc_data["nr_size"], n_coords=n_coords)
        encoder.load_state_dict(enc_data["state_dict"])
        encoder = encoder.to(location)
        return encoder
    return None

"""
def run_hg_split(hg, data, location):
    hg.eval()

    images = []
    gts = []
    preds = []

    for batch in data:
        image = move2device(batch['image'].float(), location)
        gt = move2device(batch['landmarks'].float(), location)

        with torch.no_grad():
            coords, heatmaps, before_stn, after_stn, affine_params, immediate_coords, var, unnormalized_heatmaps = hg(image)

        images.append(image)
        gts.append(gt)
        preds.append(coords)

    images = torch.cat(images)
    gts = torch.cat(gts)
    preds = torch.cat(preds)

    eval = evaluate(preds, gts)

    return {
        "images": images,
        "gt": gts,
        "prediction": preds,
        "evaluation": eval
    }


def run_hg(hg, easy, hard, location):
    return {
        "easy": run_hg_split(hg, easy, location),
        "hard": run_hg_split(hg, hard, location)
    }
"""


def run_e2e_split(e2e, data, location):
    e2e.eval()

    images = []
    gts = []
    hg_preds = []
    pdm_preds = []
    pdm_3d = []
    pdm_applied = []
    pdm_encoder_preds = []

    for batch in data:
        image = move2device(batch['image'].float(), location)
        gt = move2device(batch['landmarks'].float(), location)

        with torch.no_grad():
            pdm_pred, hg_pred, pdm_res_3d, pdm_zs, pdm_nr, apply_pdm, pdm_encoder_pred = e2e(image)

        images.append(image)
        gts.append(gt)
        hg_preds.append(hg_pred)
        pdm_preds.append(pdm_pred)
        pdm_3d.append(pdm_res_3d)
        pdm_applied.append(apply_pdm)
        if pdm_encoder_pred is not None:
            pdm_encoder_preds.append(pdm_encoder_pred)

    images = torch.cat(images)
    gts = torch.cat(gts)
    hg_preds = torch.cat(hg_preds)
    pdm_preds = torch.cat(pdm_preds)
    pdm_3d = torch.cat(pdm_3d)
    pdm_applied = torch.cat(pdm_applied)

    eval_hg = evaluate(hg_preds, gts)
    eval_pdm = evaluate(pdm_preds, gts)

    res = {
        "images": images,
        "gt": gts,
        "hg_pred": hg_preds,
        "pdm_pred": pdm_pred,
        "pdm_3d": pdm_3d,
        "eval_hg": eval_hg,
        "eval_pdm": eval_pdm,
        "pdm_applied": pdm_applied
    }

    if len(pdm_encoder_preds) > 0:
        pdm_encoder_preds = torch.cat(pdm_encoder_preds)
        eval_pdm_encoder = evaluate(pdm_encoder_pred, gts)
        res["pdm_encoder_pred"] = pdm_encoder_preds
        res["eval_pdm_encoder"] = eval_pdm_encoder

    return res


def run_e2e(e2e, easy, hard, location):
    return {
        "easy": run_e2e_split(e2e, easy, location),
        "hard": run_e2e_split(e2e, hard, location)
    }


def run(*, hg, pdm, data_src, location, hg_bs, encoder=None, verbose=True, random_seed=None, var_thresh=None, menpo=None):
    torch.autograd.set_detect_anomaly(True)  # This makes debugging much easier

    if location is not 'cpu':
        torch.cuda.set_device(torch.device(location))

    if random_seed is not None:
        make_deterministic(random_seed)

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    pin_memory = location != 'cpu'
    num_workers = 4

    with h5py.File(data_src, 'r') as f:
        easy_d = FaceLandmarksEasyTestData(f, transform=transform)
        hard_d = FaceLandmarksHardTestData(f, transform=transform)

    easy_loader = DataLoader(dataset=easy_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,  batch_size=len(easy_d))
    hard_loader = DataLoader(dataset=hard_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=len(hard_d))
    pipeline = E2E(hg, pdm, hg_bs, max(len(easy_d), len(hard_d)), encoder=encoder, verbose=verbose, var_thresh=var_thresh)
    e2e_results = run_e2e(pipeline, easy_loader, hard_loader, location)

    hg_results = {
        "easy68": e2e_results["easy"]["eval_hg"]["with_outline"],
        "hard68": e2e_results["hard"]["eval_hg"]["with_outline"],
        "easy49": e2e_results["easy"]["eval_hg"]["without_outline"],
        "hard49": e2e_results["hard"]["eval_hg"]["without_outline"]
    }

    pdm_results = {
        "easy68": e2e_results["easy"]["eval_pdm"]["with_outline"],
        "hard68": e2e_results["hard"]["eval_pdm"]["with_outline"],
        "easy49": e2e_results["easy"]["eval_pdm"]["without_outline"],
        "hard49": e2e_results["hard"]["eval_pdm"]["without_outline"]
    }

    if encoder is not None:
        pdm_encoder_results = {
            "easy68": e2e_results["easy"]["eval_pdm_encoder"]["with_outline"],
            "hard68": e2e_results["hard"]["eval_pdm_encoder"]["with_outline"],
            "easy49": e2e_results["easy"]["eval_pdm_encoder"]["without_outline"],
            "hard49": e2e_results["hard"]["eval_pdm_encoder"]["without_outline"]
        }
    else:
        pdm_encoder_results = {k: 10000000.0 for k in ["easy68", "hard68", "easy49", "hard49"]}

    if menpo is not None:
        with h5py.File(args.menpo, 'r') as f:
            menpo_d = Menpo(f, transform=transform)
            menpo_loader = DataLoader(dataset=menpo_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=len(menpo_d))

            pipeline = E2E(hg, pdm, hg_bs, len(menpo_d), encoder=encoder, verbose=verbose, var_thresh=var_thresh)
            menpo_res = run_e2e_split(pipeline, menpo_loader, location)
            menpo_gt = menpo_res["gt"]
            menpo_hg_pred = menpo_res["hg_pred"]
            menpo_pdm_pred = menpo_res["pdm_pred"]

            menpo_hg_error = evaluate_menpo(menpo_hg_pred, menpo_gt)
            menpo_pdm_error = evaluate_menpo(menpo_pdm_pred, menpo_gt)

            hg_results["menpo68"] = menpo_hg_error[0]
            hg_results["menpo49"] = menpo_hg_error[1]
            pdm_results["menpo68"] = menpo_pdm_error[0]
            pdm_results["menpo49"] = menpo_pdm_error[1]
    else:
        hg_results["menpo68"] = 10000000.0
        hg_results["menpo49"] = 10000000.0
        pdm_results["menpo68"] = 10000000.0
        pdm_results["menpo49"] = 10000000.0


    res = {
        "hg": hg_results,
        "pdm": pdm_results,
        "pdm_encoder": pdm_encoder_results,
        "gt": {
            "easy": e2e_results["easy"]["gt"],
            "hard": e2e_results["hard"]["gt"]
        },
        "hg_pred": {
            "easy": e2e_results["easy"]["hg_pred"],
            "hard": e2e_results["hard"]["hg_pred"]
        },
        "pdm_pred": {
            "easy": e2e_results["easy"]["pdm_pred"],
            "hard": e2e_results["hard"]["pdm_pred"]
        },
        "pdm_3d": {
            "easy": e2e_results["easy"]["pdm_3d"],
            "hard": e2e_results["hard"]["pdm_3d"]
        },
        "pdm_applied": {
            "easy": e2e_results["easy"]["pdm_applied"],
            "hard": e2e_results["hard"]["pdm_applied"]
        }
    }

    if "pdm_encoder_pred" in e2e_results["easy"]:
        res["pdm_encoder_pred"] = {
            "easy": e2e_results["easy"]["pdm_encoder_pred"],
            "hard": e2e_results["hard"]["pdm_encoder_pred"]
        }
    return res


def load_and_run(*, hg_src,
                    pdm_src,
                    data_src,
                    gpu_id,
                    encoders=None,
                    verbose=True,
                    random_seed=None,
                    pdm_configurator=None,
                    var_thresh=None,
                    menpo=None):
    location = 'cpu' if gpu_id < 0 else "cuda:%d" % gpu_id

    hg_src = os.path.expanduser(hg_src)
    pdm_src = os.path.expanduser(pdm_src)
    data_src = os.path.expanduser(data_src)

    hg, hg_config = load_hg(hg_src, location)
    pdm, pdm_config = load_pdm(pdm_src, location)
    pdm.verbose = True
    pdm.print_losses = False

    n_lm_hg = int(hg_config.get("n_lm", 68))
    n_lm_pdm = 49 if pdm_config["is_49lm"] else 68
    if n_lm_hg != n_lm_pdm:
        raise RuntimeError("number of LMs not the same for HG (%d) and PDM (%d)" % (n_lm_hg, n_lm_pdm))

    encoder_src = None
    n_coords = int(49 if pdm_config["is_49lm"] else 68)
    if encoders is None:
        encoder_src = None
    elif isinstance(encoders, str):  # legacy
        encoder_src = encoders
    elif isinstance(encoders, dict):  # dict with an encoder for 49 and 68 LM, choose which one fits
        encoder_src = encoders[n_coords]
    else:
        raise ValueError("No idea what to do with encoders: %s" % str(encoders))
    encoder = load_encoder(encoder_src, location, n_coords=n_coords)

    if pdm_configurator is not None:
        pdm_configurator(pdm)

    res = run(hg=hg,
              pdm=pdm,
              data_src=data_src,
              location=location,
              hg_bs=hg_config["batch_size"],
              encoder=encoder,
              verbose=verbose,
              random_seed=random_seed,
              var_thresh=var_thresh,
              menpo=menpo)
    return res, hg_config, pdm_config


def get_pdm_configurator(config):
    def pdm_configurator(pdm):
        pdm.verbose = True
        pdm.print_losses = False
        pdm.configure_for_test(config)
    return pdm_configurator


def augment_pdm_options(parser):
    parser.add_argument('--epochs', type=int, default=500, help="epochs for PDM testing")
    parser.add_argument('--optimizer', type=json.loads, default='{"name" : "adam", "lr" : 0.001}', help="optimizer settings for PDM shape")
    parser.add_argument('--scheduler', type=json.loads, default=None, help="scheduler settings for PDM shape")
    parser.add_argument('--conf_params', default=None, type=float, nargs=2, help="a,b of var -> conf transform")
    parser.add_argument('--test_reduction', default="sum", choices=["sum", "mean"],
                        help="The reduction used to calculate the loss when running PDM in test mode. "
                             "This should only be set to 'mean' when trying to reproduce older results because"
                             "'sum' is numerically more stable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training PDM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hg", type=str, help="Path to pretrained hourglass (.torch)")
    parser.add_argument("pdm", type=str, help="Path to pretrained PDM (.torch)")
    parser.add_argument("data", type=str, help="all_data_valid_w_profile_pts.h5")
    parser.add_argument('--encoder', type=str, help="Path to PDM encoder")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=None, help="Set random seed")
    parser.add_argument("--variance_threshold", type=float, default=None, help="Apply PDM only to samples with max variance > threshold")
    parser.add_argument("--menpo", type=str, help="path to h5")
    augment_pdm_options(parser)
    args = parser.parse_args()

    results = load_and_run(hg_src=args.hg,
                           pdm_src=args.pdm,
                           data_src=args.data,
                           gpu_id=args.gpu,
                           encoders=args.encoder,
                           pdm_configurator=get_pdm_configurator(vars(args)),
                           verbose=True,
                           random_seed=args.random_seed,
                           var_thresh=args.variance_threshold,
                           menpo=args.menpo)

    for k,v in results[0].items():
        if k in ["hg", "pdm", "pdm_encoder"]:
            print(k, v)
