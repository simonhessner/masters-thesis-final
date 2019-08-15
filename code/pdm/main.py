from common.util import load_config, mkdir_if_not_exists
from common.trainer import BaseModelTrainer
import os
import h5py
import argparse
import torch
from pdm.pdmcore import PDM
import json
from common.util import make_deterministic
from common.metrics import evaluate
import numpy as np


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
            self.loss_log = {"train": [], "test": []}
            self.metrics_log = {}

            self.category_calculator = {
            "e49": lambda metrics: metrics["e49"],
            "h49": lambda metrics: metrics["h49"],
            "e68": lambda metrics: metrics["e68"],
            "h68": lambda metrics: metrics["h68"],
            "49":  lambda metrics: (metrics["e49"] + metrics["h49"])/2,
            "68":  lambda metrics: (metrics["e68"] + metrics["h68"])/2,
            "e":   lambda metrics: (metrics["e49"] + metrics["e68"])/2,
            "h":   lambda metrics: (metrics["h49"] + metrics["h68"])/2,
            "all": lambda metrics: (metrics["e49"] + metrics["h49"] + metrics["e68"] + metrics["h68"])/4
        }
        self.categories = self.category_calculator.keys()
        self.best_epoch = {k:0 for k in self.categories}
        self.lowest_error = {k:np.Inf for k in self.categories}

    @staticmethod
    def create_net(config):
        if "shuffle" not in config:
            config["shuffle"] = False

        if "activation" not in config:
            config["activation"] = "relu"

        if "last_linear" not in config:
            config["last_linear"] = False

        if "is_49lm" not in config:
            config["is_49lm"] = False

        if "net_scheduler" not in config:
            config["net_scheduler"] = [config["net_scheduler_step_size"], config["net_scheduler_factor"]]
            del config["net_scheduler_step_size"]
            del config["net_scheduler_factor"]

        #print(config)
        #exit()

        pdm = PDM(layers=config["layers"],
                  activation=config["activation"],
                  last_linear=config["last_linear"],
                  train_epochs=config["epochs_train"],
                  test_epochs=config["epochs_test"],
                  bs=config["bs"],
                  is_68_lm_pdm=(not config["is_49lm"]),
                  lr_shape=config["lr_shape"],
                  lr_net=config["lr_net"],
                  multiplicative_confidence=config["multiplicative_confidence"],
                  net_scheduler=config["net_scheduler"],
                  shuffle=config["shuffle"],
                  conf_params=config["conf_params"],
                  loss=config["loss"],
                  noise_std=config["noise_std"],
                  shape_init=config["shape_init"],
                  target=config["model_dir"])
        pdm.config = config
        return pdm

    def receive_pdm_output(self, data):
        if data["mode"] == "train":
            metrics = data["metrics"]
            epoch = data["epoch"]
            pdm = data["pdm"]

            if metrics is not None:
                self.metrics_log[epoch] = metrics

                for category, calculator in self.category_calculator.items():
                    error = calculator(metrics)

                    if error < self.lowest_error[category] and error < 1000:  # 100000 is the error for with outline when HG only has 49LM
                        self.lowest_error[category] = error
                        self.best_epoch[category] = epoch

                        pdm_target = os.path.join(self.model_dir, "%d_best_%s.torch" % (self.config["config_id"], category))
                        pdm.save_pdm(epoch, pdm_target)

            if epoch % 25 == 1:
                if metrics is None:
                    print("GPU %d.%d | Config %04d | %d LM | Epoch %0.4d/%0.4d | Loss %0.5f" % (self.gpu_id, self.sub_gpu_id, self.config["config_id"], 49 if self.config["is_49lm"] else 68, data["epoch"], self.config["epochs_train"], data["loss"]))
                else:
                    print("GPU %d.%d | Config %04d | Epoch %0.4d/%0.4d | Loss %0.5f | easy (49) %05.2f [best %05.2f] | easy (68) %05.2f [best %05.2f] | hard (49) %05.2f [best %05.2f] | hard (68) %05.2f [best %05.2f]" % (
                        self.gpu_id,
                        self.sub_gpu_id,
                        self.config["config_id"],
                        data["epoch"],
                        self.config["epochs_train"],
                        data["loss"],
                        metrics["e49"],
                        self.lowest_error["e49"],
                        metrics["e68"],
                        self.lowest_error["e68"],
                        metrics["h49"],
                        self.lowest_error["h49"],
                        metrics["h68"],
                        self.lowest_error["h68"]
                    ))

            self.loss_log["train"].append(data["loss"])
        #else: # ignore testing

    #@profile
    def run(self):
        self.config["model_dir"] = self.model_dir

        make_deterministic(self.config['random_seed'])

        pdm = ModelTrainer.create_net(self.config)
        self.to_gpu(pdm)
        pdm.verbose = not self.is_gridsearch
        pdm.listener = self.receive_pdm_output

        dt = h5py.File(self.data, "r")
        data_tr = self.to_gpu(torch.tensor(dt["300W"]["train_y"], dtype=torch.float32))
        data_te = self.to_gpu(torch.tensor(dt["300W"]["test_y"],  dtype=torch.float32))

        if self.config["add_multipie"]:
            tmp = self.to_gpu(torch.tensor(dt["multipie"]["train_y"], dtype=torch.float32))
            data_tr = torch.cat((data_tr, tmp))

        #print("train", data_tr.shape)
        #print("test", data_te.shape)
        #exit()

        zs_tr, nr_tr, loss_tr = pdm.train(data=data_tr)
        train_reconstructed, _ = pdm.forward(zs_tr, nr_tr)

        zs_te, nr_te, loss_te, *_ = pdm.test(data=data_te, confidence=None)
        test_reconstructed, _ = pdm.forward(zs_te, nr_te)

        target_file = os.path.join(self.result_dir, "zs_and_nr_%d.json" % self.config["config_id"])

        json.dump({
            "train": {
                "zs": zs_tr.detach().cpu().numpy().tolist(),
                "nr": nr_tr.detach().cpu().numpy().tolist(),
                "reconstructed": train_reconstructed.detach().cpu().numpy().tolist(),
                "coords": data_tr.detach().cpu().numpy().tolist()
            },
            "test": {
                "zs": zs_te.detach().cpu().numpy().tolist(),
                "nr": nr_te.detach().cpu().numpy().tolist(),
                "reconstructed": test_reconstructed.detach().cpu().numpy().tolist(),
                "coords": data_te.detach().cpu().numpy().tolist()
            }
        }, open(target_file, "w"))

        pdm.save_pdm(pdm.train_epochs, os.path.join(self.model_dir, "final_pdm_%d.torch" % self.config["config_id"]))

        # TODO train ENCODERS DIRECTLY HERE

        if self.is_gridsearch:
            last_train_loss = self.loss_log["train"][-1]
            lowest_train_loss = min(self.loss_log["train"])
            best_train_epoch = min([i for i in range(len(self.loss_log["train"])) if self.loss_log["train"][i] == lowest_train_loss])

            train_error = evaluate(train_reconstructed, data_tr)
            test_error = evaluate(test_reconstructed, data_te)

            #print(train_error, test_error)

            best_epochs = {"best_%s_epoch" % k: v for k, v in self.best_epoch.items()}
            best_errors = {"best_%s" % k: v for k, v in self.lowest_error.items()}

            return {
                **self.config,
                "last_train_loss" : last_train_loss,
                "lowest_train_loss" : lowest_train_loss,
                "best_train_epoch" : best_train_epoch,
                "metrics_log" : self.metrics_log,
                **best_epochs,
                **best_errors,
                "train_error_49": train_error["without_outline"],
                "train_error_68": train_error["with_outline"],
                "test_error_49": test_error["without_outline"],
                "test_error_68": test_error["with_outline"]
            }
        else:
            # evaluate PDM
            metrics = pdm.eval_on_alpha_hg()
            print(metrics["easy_metrics_last"])
            print(metrics["hard_metrics_last"])


if __name__=="__main__":
    PDM.ALPHA_HG_SOURCE_49 = "../" + PDM.ALPHA_HG_SOURCE_49
    PDM.ALPHA_HG_SOURCE_68 = "../" + PDM.ALPHA_HG_SOURCE_68
    parser = argparse.ArgumentParser(description='Training PDM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", type=str, help="all_data_valid_w_profile_pts.h5")
    parser.add_argument('output_directory',
                        type=str,
                        help="Where to store results")
    parser.add_argument('--layers',
                        type=int,
                        default=[128],
                        nargs="+",
                        metavar='N',
                        help='Layer configuration of the model, first layer\'s dim is zdim')
    parser.add_argument('--activation',
                        type=str,
                        choices=["relu", "tanh", "sigmoid"],
                        default="relu",
                        help="PDM activation function")
    parser.add_argument('--last_linear', default=False, action="store_true", help="Use activation for last layer or not (=linear)?")
    parser.add_argument('--is_49lm', default=False, action='store_true', help="Train on 49 instead 68 LM")
    parser.add_argument('--epochs_train',
                        type=int,
                        default=500,
                        metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--epochs_test',
                        type=int,
                        default=500,
                        metavar='N',
                        help='number of epochs to test')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.001,
                        metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_shape',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate')
    parser.add_argument('--multiplicative_confidence',
                        action='store_true',
                        default=False,
                        help="Weather to use confidence 'dropout'")
    parser.add_argument('--net_scheduler', type=float, nargs=2, default=None, help="Format: step_size factor")
    parser.add_argument('--bs',
                        type=int,
                        default=32,
                        metavar='N',
                        help='input batch size for training')
    parser.add_argument("--shuffle",
                        action="store_true",
                        default=False,
                        help="Shuffle in each epoch")
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        metavar="ID",
                        help="GPU ID, -1 to use CPU")
    parser.add_argument('--random_seed',
                        type=int,
                        default=0)
    args = vars(parser.parse_args())
    args["config_id"] = 0 # for the non-gridsearch case

    args["conf_params"] = [0.1, 0.1]
    args["loss"] = ["wing", 5.0, 0.5]
    args["noise_std"] = 0.00000000001
    args["shape_init"] = ("uniform", 10e-2)


    trainer = ModelTrainer(args, args["output_directory"], args["gpu"], 1, args["data"], gridsearch=False)
    trainer.run()
    print()
