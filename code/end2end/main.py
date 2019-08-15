from common.util import load_config, mkdir_if_not_exists
from common.trainer import BaseModelTrainer
import os
import argparse
import torch
import json
import numpy as np
import torch.backends.cudnn as cudnn
from pe_hourglass.main import ModelTrainer as HGModelTrainer
from pdm.main import ModelTrainer as PDMModelTrainer
from datasets.facial_landmarks import FaceLandmarksEasyTestData, FaceLandmarksTrainingData, FaceLandmarksHardTestData
from common.transforms import ImageTransform, RandomHorizontalFlip, ImageAndLabelTransform
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import h5py
from common.util import make_deterministic
from end2end.run_all import run as run_e2e
import matplotlib.pyplot as plt

from pdm.pdmcore import PDM
PDM.ALPHA_HG_SOURCE = "../" + PDM.ALPHA_HG_SOURCE


class ModelTrainer(BaseModelTrainer):
    def __init__(self, config, output_dir, gpu_id, sub_gpu_id, data, gridsearch):
        super(ModelTrainer, self).__init__(gpu_id, sub_gpu_id)

        if isinstance(config, dict):
            self.config = config
        else:
            self.config = load_config(config, verbose=False)
        self.data = data

        self.output_dir = output_dir
        mkdir_if_not_exists(self.output_dir)

        self.model_dir = os.path.join(self.output_dir, "models")
        mkdir_if_not_exists(self.model_dir)

        self.result_dir = os.path.join(self.output_dir, "results")
        mkdir_if_not_exists(self.result_dir)

        self.plot_dir = os.path.join(self.output_dir, "plots")
        mkdir_if_not_exists(self.plot_dir)

        self.is_gridsearch = gridsearch
        if self.is_gridsearch:
            self.loss_log = []
            self.l2d_log = []
            self.gts = []
            self.hg_coords_log = []

    def receive_pdm_output(self, data):
        if data["mode"] != "e2e":
            return

        pdm = data["pdm"]
        hg = data["hg"]
        epoch = data["epoch"]
        self.loss_log.append(data["loss"])
        self.l2d_log.append(data["l2d_coords"][:4])  # Save space by only appending 4
        self.hg_coords_log.append(data["hg_coords"][:4])
        if len(self.gts) == 0:
            self.gts = data["gt_coords"][:4]

        epochs = self.config["training_schedule"][0]
        phase2_start = self.config["training_schedule"][1]
        hg_train_frequency = self.config["training_schedule"][2]

        if epoch % 50 == 0:
            print("GPU %d.%d | Config %04d | Mode %s | Epoch %d/%d  | Loss %0.5f" % (self.gpu_id, self.sub_gpu_id, self.config["config_id"], data["mode"], data["epoch"], epochs, data["loss"]))


    def load_hg(self, path, location):
        data = torch.load(path, map_location=location)
        state_dict = data['state_dict']
        config = data['config']

        hg = HGModelTrainer.create_net(config, verbose=False)
        hg.load_state_dict(state_dict)
        hg.eval()

        hg = hg.to(location)

        return hg, config

    def load_pdm(self, path, location):
        data = torch.load(path, map_location=location)
        state_dict = data['state_dict']
        config = data['config']

        pdm = PDMModelTrainer.create_net(config, load_alpha_hg=False)
        pdm.model.load_state_dict(state_dict)
        pdm = pdm.to(location)
        return pdm, config

    def run(self):
        torch.autograd.set_detect_anomaly(True)  # This makes debugging much easier

        self.config["model_dir"] = self.model_dir

        make_deterministic(self.config['random_seed'])

        location = 'cpu' if self.gpu_id is None else "cuda:%d" % self.gpu_id
        if location is not 'cpu':
            # This fixes the problem that pytorch is always allocating memory on GPU 0 even if this is not included
            # in the list of GPUs to use
            torch.cuda.set_device(torch.device(location))

            # cudnn.benchmark improves training speed when input sizes do not change
            # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
            # It selects the best algorithms as the training iterates over the dataset
            #cudnn.benchmark = True # but it can cause determinism problems, so disable

        hg, hg_config = self.load_hg(self.config["initial_hg"], location)
        pdm, pdm_config = self.load_pdm(self.config["initial_pdm"], location)

        pdm.verbose = not self.is_gridsearch
        pdm.print_losses = False
        pdm.listener = self.receive_pdm_output

        normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
        normTransform = transforms.Normalize(normMean, normStd)

        jitterTransform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

        transform = transforms.Compose([
            ImageTransform(transforms.ToPILImage()),
            ImageTransform(jitterTransform),
            ImageAndLabelTransform(RandomHorizontalFlip()),
            ImageTransform(transforms.ToTensor()),
            ImageTransform(normTransform)
        ])

        bs = self.config["bs"]
        pin_memory = location != 'cpu'
        num_workers = 8

        with h5py.File(self.config["data"], 'r') as f:
            train_d = FaceLandmarksTrainingData(f, transform=transform)
            train_loader = DataLoader(dataset=train_d, shuffle=self.config["shuffle"], num_workers=num_workers, pin_memory=pin_memory, batch_size=bs)

        results_before = run_e2e(hg, pdm, self.config["data"], location, self.config["bs"], verbose=True)
        if not self.is_gridsearch:
            print("Before training")
            for model, res in results_before.items():
                print(model, res)

        zs, nr, losses = pdm.end2end_training(hg=hg,
                                              data_loader=train_loader,
                                              hg_opt_config=self.config["hg_optimizer"],
                                              pdm_weight_opt_config=self.config["pdm_weight_optimizer"],
                                              pdm_shape_opt_config=self.config["pdm_shape_optimizer"],
                                              training_schedule=self.config["training_schedule"],
                                              detach_confidence=self.config["detach_confidence"])

        plot_path = os.path.join(self.plot_dir, "losses_%d.png" % self.config["config_id"])
        if not self.is_gridsearch: print("save plot to %s" % plot_path)
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set(xlabel='epoch', ylabel='loss', title='loss per epoch')
        ax.grid()
        fig.savefig(plot_path)

        if not self.is_gridsearch: print("save HG")
        torch.save({
            'model': 'pe_hourglass',
            'state_dict': hg.state_dict(),
            'config': hg_config
        }, os.path.join(self.model_dir, "%d_hg_e2e.torch" % self.config["config_id"]))

        if not self.is_gridsearch: print("save PDM")
        pdm.save_pdm(pdm.train_epochs, os.path.join(self.model_dir, "%d_pdm_e2e.torch" % self.config["config_id"]))

        results_after = run_e2e(hg, pdm, self.config["data"], location, self.config["bs"], verbose=False)

        if not self.is_gridsearch:
            print("Before training")
            for model, res in results_before.items():
                print(model, res)

            print("After training")
            for model, res in results_after.items():
                print(model, res)

        if self.is_gridsearch:
            logpath = os.path.join(self.result_dir, "%d_log.json" % self.config["config_id"])
            json.dump({
                "gt": self.gts,
                "l2d": self.l2d_log,
                "hg": self.hg_coords_log,
                "losses": self.loss_log
            }, open(logpath, "w"))

            return {
                **self.config,
                "min_loss": min(self.loss_log),
                "last_loss" : self.loss_log[-1],
                "hg_before_easy_with" : results_before["hg"]["easy_woutline"],
                "hg_before_easy_without": results_before["hg"]["easy_noutline"],
                "hg_before_hard_with": results_before["hg"]["hard_woutline"],
                "hg_before_hard_without": results_before["hg"]["hard_noutline"],
                "pdm_before_easy_with": results_before["pdm"]["easy_woutline"],
                "pdm_before_easy_without": results_before["pdm"]["easy_noutline"],
                "pdm_before_hard_with": results_before["pdm"]["hard_woutline"],
                "pdm_before_hard_without": results_before["pdm"]["hard_noutline"],
                "hg_after_easy_with": results_after["hg"]["easy_woutline"],
                "hg_after_easy_without": results_after["hg"]["easy_noutline"],
                "hg_after_hard_with": results_after["hg"]["hard_woutline"],
                "hg_after_hard_without": results_after["hg"]["hard_noutline"],
                "pdm_after_easy_with": results_after["pdm"]["easy_woutline"],
                "pdm_after_easy_without": results_after["pdm"]["easy_noutline"],
                "pdm_after_hard_with": results_after["pdm"]["hard_woutline"],
                "pdm_after_hard_without": results_after["pdm"]["hard_noutline"],
            }


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training PDM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("initial_hg", type=str, help="Path to pretrained hourglass (.torch)")
    parser.add_argument("initial_pdm", type=str, help="Path to pretrained PDM (.torch)")
    parser.add_argument("data", type=str, help="all_data_valid_w_profile_pts.h5")
    parser.add_argument('output_directory', type=str, help="Where to store results")
    parser.add_argument('--bs', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help="epochs for end2end training")
    parser.add_argument('--shape_init_epochs', type=int, default=1500, help="epoch where phase 2 begins")
    parser.add_argument('--hg_train_frequency', type=int, default=100, help="Train HG every N epochs, otherwise run PDM in test mode")
    parser.add_argument('--hg_pdm_together', default=False, action='store_true', help="Weather to train HG and PDM in the same epoch or only HG")
    parser.add_argument('--shuffle', default=False, action='store_true', help="Shuffle training data")
    parser.add_argument('--hg_optimizer', type=json.loads, default='{"name" : "sgd", "lr" : 0.001}', help="optimizer settings for HG")
    parser.add_argument('--pdm_weight_optimizer', type=json.loads, default='{"name" : "sgd", "lr" : 0.001}', help="optimizer settings for PDM weights")
    parser.add_argument('--pdm_shape_optimizer', type=json.loads, default='{"name" : "sgd", "lr" : 0.1}', help="optimizer settings for PDM shape")
    parser.add_argument('--detach_confidence', default=False, action='store_true', help="No gradient through confidence")
    parser.add_argument('--gpu', type=int, default=0, metavar="ID", help="GPU ID, -1 to use CPU")
    parser.add_argument('--random_seed', type=int, default=1)
    args = vars(parser.parse_args())


    args["config_id"] = 0
    args["training_schedule"] = (args["epochs"], args["shape_init_epochs"], args["hg_train_frequency"], args["hg_pdm_together"])

    print("\ngiven parameters:")
    for k,v in args.items():
        print(k, v)
    print("\n")

    trainer = ModelTrainer(args, args["output_directory"], args["gpu"], 0, args["data"], gridsearch=False)
    trainer.run()
    print()
