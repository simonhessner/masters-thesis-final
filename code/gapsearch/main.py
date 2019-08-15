from end2end.run_all import load_and_run
from common.util import load_config
from common.trainer import BaseModelTrainer
import argparse
import torch
import os
from common.util import make_deterministic, mkdir_if_not_exists
import time
import random
from end2end.run_all import augment_pdm_options
import json


class ModelTrainer(BaseModelTrainer):
    def __init__(self, config, output_dir, gpu_id, sub_gpu_id, data, gridsearch):  # output directory is not really used
        super(ModelTrainer, self).__init__(gpu_id, sub_gpu_id)

        if isinstance(config, dict):
            self.config = config
        else:
            self.config = load_config(config, verbose=False)
        self.data = data

        self.is_gridsearch = gridsearch

    def configure_pdm(self, pdm):
        pdm.print_losses = False
        pdm.verbose = not self.is_gridsearch
        pdm.configure_for_test(self.config)

    def run(self):
        torch.autograd.set_detect_anomaly(True)  # This makes debugging much easier
        make_deterministic(self.config['random_seed'])

        encoders = None
        if self.config["encoder"]:
            # This assumes that an encoder has already been trained for the PDM
            # Example: pdm path is my/dir/models/pdm_4.torch
            # Then the encoder is loaded from my/dir/encoders/encoder_4.torch (if it does not exists, the code crashes)
            pdm_filename = os.path.basename(self.config["pdm"])
            if "final" in pdm_filename:
                pdm_id = int(pdm_filename.split(".")[0].split("_")[-1])
            else:
                pdm_id = int(pdm_filename.split("_")[0])
            encoders = {
                49: os.path.join(os.path.dirname(os.path.dirname(self.config["pdm"])), "encoders", "encoder_49_%d.torch" % pdm_id),
                68: os.path.join(os.path.dirname(os.path.dirname(self.config["pdm"])), "encoders", "encoder_68_%d.torch" % pdm_id),
            }

        if not self.is_gridsearch:
            print("encoder", encoders)

        if "prediction_target" in self.config and self.config["prediction_target"] is not None:
            pred_target = self.config["prediction_target"]
            pred_target_dir = os.path.dirname(pred_target)
            mkdir_if_not_exists(pred_target_dir)
        else:
            pred_target = None

        success = False
        tries = 0
        maxtries = 75
        while not success:
            tries += 1
            try:
                res, hg_config, pdm_config = load_and_run(hg_src=self.config["hg"],
                                                          pdm_src=self.config["pdm"],
                                                          data_src=self.data,
                                                          gpu_id=self.gpu_id,
                                                          random_seed=self.config["random_seed"],
                                                          pdm_configurator=self.configure_pdm,
                                                          verbose=not self.is_gridsearch,
                                                          var_thresh=self.config["variance_threshold"],
                                                          encoders=encoders)
                success = True
            except RuntimeError as e:
                txt = str(e)
                if "out of memory" in txt:
                    if tries <= maxtries:
                        waittime = tries * random.randint(1,5)
                        print("ERROR! There was a OOM error, wait %d seconds and try again. Try nr. %d" % (waittime, tries))
                        time.sleep(waittime)
                    else:
                        print("ERROR! maxtries (%d) exceeded" % maxtries)
                        raise e
                else:
                    raise e

        results = {
            "hg_easy49": res["hg"]["easy49"],
            "hg_hard49": res["hg"]["hard49"],
            "hg_easy68": res["hg"]["easy68"],
            "hg_hard68": res["hg"]["hard68"],
            "pdm_easy49": res["pdm"]["easy49"],
            "pdm_hard49": res["pdm"]["hard49"],
            "pdm_easy68": res["pdm"]["easy68"],
            "pdm_hard68": res["pdm"]["hard68"],
            "pdm_encoder_easy49": res["pdm_encoder"]["easy49"],
            "pdm_encoder_hard49": res["pdm_encoder"]["hard49"],
            "pdm_encoder_easy68": res["pdm_encoder"]["easy68"],
            "pdm_encoder_hard68": res["pdm_encoder"]["hard68"],
            "easy49_factor": res["hg"]["easy49"] / res["pdm"]["easy49"],
            "hard49_factor": res["hg"]["hard49"] / res["pdm"]["hard49"],
            "easy68_factor": res["hg"]["easy68"] / res["pdm"]["easy68"],
            "hard68_factor": res["hg"]["hard68"] / res["pdm"]["hard68"],
            "enc_easy49_factor": res["hg"]["easy49"] / res["pdm_encoder"]["easy49"] if self.config["encoder"] else 0.0,
            "enc_hard49_factor": res["hg"]["hard49"] / res["pdm_encoder"]["hard49"] if self.config["encoder"] else 0.0,
            "enc_easy68_factor": res["hg"]["easy68"] / res["pdm_encoder"]["easy68"] if self.config["encoder"] else 0.0,
            "enc_hard68_factor": res["hg"]["hard68"] / res["pdm_encoder"]["hard68"] if self.config["encoder"] else 0.0
        }

        print("Config: %d | factor e49: %0.4f | factor h49: %0.4f | factor e68: %0.4f | factor h68: %0.4f" % (self.config["config_id"], results["easy49_factor"], results["hard49_factor"], results["easy68_factor"], results["hard68_factor"]))
        if self.is_gridsearch:
            return {
                **self.config,
                **results
            }
        else:
            for k,v in results.items():
                print(k, v)

        if pred_target:
            output = {
                "meta": {
                    "hg_model":   self.config["hg"],
                    "pdm_model":  self.config["pdm"],
                    "hg_config":  hg_config,
                    "pdm_config": pdm_config,
                    "gapsearch_config": self.config
                },
                "results": results,
                "predictions": {
                    "easy": {
                        "gt": res["gt"]["easy"].cpu().detach().numpy().tolist(),
                        "pdm_pred": res["pdm_pred"]["easy"].cpu().detach().numpy().tolist(),
                        "hg_pred": res["hg_pred"]["easy"].cpu().detach().numpy().tolist(),
                        "pdm_3d": res["pdm_3d"]["easy"].cpu().detach().numpy().tolist()
                    },
                    "hard": {
                        "gt": res["gt"]["hard"].cpu().detach().numpy().tolist(),
                        "pdm_pred": res["pdm_pred"]["hard"].cpu().detach().numpy().tolist(),
                        "hg_pred": res["hg_pred"]["hard"].cpu().detach().numpy().tolist(),
                        "pdm_3d": res["pdm_3d"]["hard"].cpu().detach().numpy().tolist()
                    }
                }
            }

            if "pdm_encoder_pred" in res:
                output["predictions"]["easy"]["pdm_encoder_pred"] = res["pdm_encoder_pred"]["easy"].cpu().detach().numpy().tolist()
                output["predictions"]["hard"]["pdm_encoder_pred"] = res["pdm_encoder_pred"]["hard"].cpu().detach().numpy().tolist()

            json.dump(output, open(pred_target, "w"), indent=2)
            print("Predictions written to", pred_target)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training PDM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hg", type=str, help="Path to pretrained hourglass (.torch)")
    parser.add_argument("pdm", type=str, help="Path to pretrained PDM (.torch)")
    parser.add_argument("data", type=str, help="all_data_valid_w_profile_pts.h5")
    parser.add_argument('--encoder', default=False, action="store_true", help="Use pre-trained encoder")
    parser.add_argument('--gpu', type=int, default=0, metavar="ID", help="GPU ID, -1 to use CPU")
    parser.add_argument('--random_seed', type=int, default=0, help="random seed")
    parser.add_argument('--variance_threshold', default=None, type=float, help="Threshold for PDM appliance")
    augment_pdm_options(parser)
    parser.add_argument('--prediction_target', default=None, type=str, help="json file where to write predictions")

    args = vars(parser.parse_args())

    args["config_id"] = 0

    trainer = ModelTrainer(args, None, args["gpu"], 0, args["data"], gridsearch=False)
    trainer.run()
    print()
