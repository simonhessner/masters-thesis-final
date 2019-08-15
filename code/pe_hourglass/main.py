import argparse
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import time
import dsntnn
import numpy as np
import json

from datasets.facial_landmarks import FaceLandmarksTrainingData, FaceLandmarksEasyTestData, FaceLandmarksHardTestData, FaceLandmarksAllTestData
from common.util import load_config, mkdir_if_not_exists, LossException, print_info, make_deterministic
from common.util import count_parameters
from common.metrics import benchmark, WingLoss
from common.transforms import RandomHorizontalFlip, ImageTransform, ImageAndLabelTransform, RandomRotation, NormalizeRotation
from common.early_stopping import EarlyStopping
from common.trainer import BaseModelTrainer
from pe_hourglass.hourglass import HourglassStack
from pe_hourglass.network import LinearRegressor, TanhRegressor, HourglassPipeline, DSNTRegressor

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class ModelTrainer(BaseModelTrainer):
    def __init__(self, config, output_dir, gpu_id, sub_gpu_id, data, gridsearch):
        super(ModelTrainer, self).__init__(gpu_id, sub_gpu_id)

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

        self.is_gridsearch = gridsearch  # TODO use this to print verbose

    def evaluate_model(self, val_loader, net, loss_function):
        net.eval()  # Switch to eval mode
        val_loss = 0
        test_sample_count = 0

        with torch.no_grad():
            for sample in (val_loader):
                x = self.to_gpu(sample['image'].float())
                y = self.to_gpu(sample['landmarks'].float())
                test_sample_count += x.shape[0]

                coords, heatmaps, *_ = net(x)
                loss = loss_function(coords, heatmaps, y)

                val_loss += loss.data.item()

        net.train()  # Revert to Train mode before resuming training
        return val_loss / test_sample_count

    # This is not the nicest code, I know. But in the DSNT paper they found that combining heatmap and coord
    # losses gives about 1% absolute improvement, so I want to use this in the dsnt case. In the other cases,
    # I am not using heatmap losses because in order to do so, they would have to be normalized which
    # I am not sure if it would affect performance negatively
    def get_loss_function(self, regression_method, loss_function_type):
        losses = {
            "l1": [nn.L1Loss],
            "mse": [nn.MSELoss],
            "l1_and_mse": [nn.L1Loss, nn.MSELoss],
            "wing_5": [lambda: WingLoss(width=5, curvature=0.5, return_sample_loss=False)],
            "wing_10": [lambda: WingLoss(width=10, curvature=1.0, return_sample_loss=False)]
        }
        coord_criterions = [loss() for loss in losses[loss_function_type]]

        def only_coord_loss(coords, _, y):
            if not isinstance(coords, list):
                coords = [coords]

            # Intermediate supervision
            coords_loss = sum([sum([criterion(c, y) for criterion in coord_criterions]) for c in coords])
            return coords_loss

        if regression_method == "dsnt":
            if self.config["heatmap_sigma"] < 0:  # No heatmap loss desired
                return only_coord_loss

            def coord_and_heatmap_loss(_coords, _heatmaps, y):
                if not isinstance(_coords, list):
                    coords = [_coords]
                    heatmaps = [_heatmaps]
                else:
                    coords = _coords
                    heatmaps = _heatmaps

                _losses = []
                for c,h in zip(coords, heatmaps): # for intermediate supervision: apply loss to all outputs
                    coords_loss = sum([criterion(c, y) for criterion in coord_criterions])
                    heatmap_losses = dsntnn.js_reg_losses(h, y, sigma_t=self.config["heatmap_sigma"])  # TODO different sigmas in each HG?
                    heatmap_loss = dsntnn.average_loss(heatmap_losses)
                    _losses.append(coords_loss + heatmap_loss)  # TODO add lambda for regularization strength??
                return sum(_losses)

            return coord_and_heatmap_loss
        else:
            return only_coord_loss

    @staticmethod
    def create_net(config, verbose=False):
        if "do_1x1_before_dsnt" not in config:
            config["do_1x1_before_dsnt"] = True

        if "intermediate_loss" not in config:
            config["intermediate_loss"] = False

        if "predict_distances_weight" not in config:
            config["predict_distances_weight"] = 0.0

        n_lm = config.get("n_lm", 68)
        assert n_lm in [49,68], "only 49 or 68 LMs allowed, %d asked" % n_lm
        hourglass = HourglassStack(n_hgs=config['n_hgs'],
                                 n_res_modules=config['n_res_modules'],
                                 n_features=config['n_features'],
                                 n_output=n_lm,
                                 hg_depth=config['hg_depth'],
                                 return_all=config["intermediate_loss"])

        regressors = {
            "tanh": lambda n: TanhRegressor(n_coords=n),
            "linear": lambda n: LinearRegressor(n_coords=n),
            "dsnt": lambda n: DSNTRegressor(n_coords=n, do_1x1=config["do_1x1_before_dsnt"])
        }
        regressor = regressors[config['regression']](n_lm)

        pipeline = HourglassPipeline(hourglass, regressor)

        if verbose:
            print("-" * 30)
            print(pipeline)
            print("-" * 30)

        return pipeline

    def run(self):
        torch.cuda.empty_cache()

        starttime = time.time()

        if self.gpu_id is not None:
            # cudnn.benchmark improves training speed when input sizes do not change
            # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
            # It selects the best algorithms as the training iterates over the dataset
            # I found no big difference between True and False, but it also doesn't hurt, so enable it
            #cudnn.benchmark = True # disable for deterministic behavior
            pass

        config = self.config
        config_id = config["config_id"]
        n_lm = config["n_lm"]

        make_deterministic(config['random_seed'])
        torch.autograd.set_detect_anomaly(True)  # This makes debugging much easier

        jitterTransform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

        # TODO store these values in h5 files
        normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
        normTransform  = transforms.Normalize(normMean, normStd)

        rot_angle = float(config['augment_rotation'])
        rotation_augmentation = RandomRotation(min_angle = -1 * rot_angle,
                                               max_angle = rot_angle,
                                               retain_scale = False,
                                               rotate_landmarks="same")

        trainTransform = transforms.Compose([
            ImageTransform(transforms.ToPILImage()),
            ImageTransform(jitterTransform),
            ImageAndLabelTransform(RandomHorizontalFlip()),
            ImageAndLabelTransform(rotation_augmentation),
            ImageTransform(transforms.ToTensor()),
            ImageTransform(normTransform)
        ])

        testTransform = transforms.Compose([
            ImageTransform(transforms.ToPILImage()),
            ImageTransform(transforms.ToTensor()),
            ImageTransform(normTransform)
        ])

        # Note: Reading takes only ~0.2s, so it is okay to do this again whenever main.py is called
        # No need to read in trainer.py and pass results here
        with h5py.File(self.data, 'r') as f:
            train_dataset = FaceLandmarksTrainingData(f, transform=trainTransform, n_lm=n_lm)
            val_dataset = FaceLandmarksAllTestData(f, transform=testTransform, n_lm=n_lm)
            easy_d = FaceLandmarksEasyTestData(f, transform=testTransform, n_lm=n_lm)
            hard_d = FaceLandmarksHardTestData(f, transform=testTransform, n_lm=n_lm)

        print("GPU %d.%d" % (self.gpu_id, self.sub_gpu_id),
              "Data: %s" % self.data,
              "Train %d Test %d" % (len(train_dataset), len(val_dataset)))

        dataloader_params = {
            'batch_size': config['batch_size'],
            'pin_memory': self.gpu_id is not None,
            'num_workers': 8
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)
        easy  = DataLoader(easy_d, shuffle=False, **dataloader_params)
        hard  = DataLoader(hard_d, shuffle=False, **dataloader_params)

        net = self.create_net(config)
        _, trainable_parameters, _ = count_parameters(net)
        self.to_gpu(net)
        net.train()  # Put net into train mode

        params = [
            {"params": net.hourglass.parameters()},
            {"params": net.regressor.parameters()},
        ]

        if config["predict_distances_weight"] > 0:
            # generate ground truth distances
            y = torch.stack([x["landmarks"] for x in train_dataset])
            bs = y.shape[0]
            n_lm = y.shape[1]
            dist_gt = torch.zeros(bs, n_lm, n_lm, 2)
            dist_gt[:,:,:,0] = y[:,:,0].view(bs,1,-1) - y[:,:,0].view(bs,-1,1)
            dist_gt[:,:,:,1] = y[:,:,1].view(bs,1,-1) - y[:,:,1].view(bs,-1,1)

        optimizer = optim.Adam(params, lr=config['lr'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['lr_scheduler_patience'], verbose=True, factor=config['lr_decay_factor'])

        early_stopping_patience = config['lr_scheduler_patience']*2 + 1
        early_stopping_max_ratio = 0.975
        should_stop = EarlyStopping(patience=early_stopping_patience, max_ratio=early_stopping_max_ratio, verbose=False)

        loss_function = self.get_loss_function(config['regression'], config['loss_function'])

        category_calculator = {
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
        categories = category_calculator.keys()
        best_epoch = {k:0 for k in categories}
        lowest_error = {k:np.Inf for k in categories}
        epoch_train_losses = []
        epoch_val_losses = []

        # Only store models that are better than these values to save storage
        storage_thresholds = {
            "e49": 2.1,
            "h49": 3.4,
            "e68": 2.7,
            "h68": 4.5
        }
        storage_thresholds["49"] = category_calculator["49"](storage_thresholds)
        storage_thresholds["68"] = category_calculator["68"](storage_thresholds)
        storage_thresholds["e"] = category_calculator["e"](storage_thresholds)
        storage_thresholds["h"] = category_calculator["h"](storage_thresholds)
        storage_thresholds["all"] = category_calculator["all"](storage_thresholds)

        loss_history = {}
        metric_history = []

        dist_loss_fct = nn.L1Loss()

        epochs = config['n_epoch']
        for epoch in range(epochs):
            epoch_start_time = time.time()

            net.train()
            epoch_train_loss = 0
            epoch_sample_count = 0

            for sample in train_loader:
                x = self.to_gpu(sample['image'].float())
                y = self.to_gpu(sample['landmarks'].float())
                if config["predict_distances_weight"] > 0:
                    indices = self.to_gpu(sample['index'])
                    dist_y = self.to_gpu(dist_gt[indices])
                epoch_sample_count += x.shape[0]

                optimizer.zero_grad()

                coords, heatmaps, var, unnormalized_heatmaps = net(x)

                loss = loss_function(coords, heatmaps, y)
                epoch_train_loss += loss.float().data.item()
                if config["normalize_loss"]:
                    if loss.detach().data.item() > 0:
                        loss = loss / loss.detach()

                if config["predict_distances_weight"] > 0:
                    bs = x.shape[0]
                    distance_pred = torch.zeros(bs, n_lm, n_lm, 2)
                    distance_pred[:,:,:,0] = coords[:,:,0].view(bs,1,-1) - coords[:,:,0].view(bs,-1,1)
                    distance_pred[:,:,:,1] = coords[:,:,1].view(bs,1,-1) - coords[:,:,1].view(bs,-1,1)
                    distance_pred = self.to_gpu(distance_pred)
                    dist_loss = dist_loss_fct(distance_pred, dist_y)
                    loss = loss + config["predict_distances_weight"] * dist_loss/dist_loss.detach()
                else:
                    dist_loss = 0

                if torch.isnan(loss):
                    print_info("ERROR! Invalid loss (nan). Aborting training for config %d in epoch %d" % (config_id, epoch))
                    raise LossException("loss was nan in config %d, epoch %d" % (config_id, epoch))
                if torch.isinf(loss):
                    print_info("ERROR! Invalid loss (inf). Aborting training for config %d in epoch %d" % (config_id, epoch))
                    raise LossException("loss was inf in config %d, epoch %d" % (config_id, epoch))

                loss.backward()
                optimizer.step()

                #### end batch

            epoch_train_loss /= epoch_sample_count  # normalize loss by images that were processed

            val_loss = self.evaluate_model(val_loader, net, loss_function)
            scheduler.step(val_loss)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(val_loss)
            loss_history[epoch] = {
                'train': epoch_train_losses[-1],
                'val': epoch_val_losses[-1]
            }

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time-epoch_start_time

            metrics = benchmark(net, easy, hard, self.gpu_id)
            all_metrics = {}
            for category, calculator in category_calculator.items():
                error = calculator(metrics)
                all_metrics[category] = error

                if error < lowest_error[category] and error < 1000:  # 100000 is the error for with outline when HG only has 49LM
                    lowest_error[category] = error
                    best_epoch[category] = epoch

                    if error < storage_thresholds[category]:
                        torch.save({
                            'model': 'pe_hourglass',
                            'epoch': epoch + 1,
                            'state_dict': net.state_dict(),
                            'val_loss': val_loss,
                            'config': config,
                            'category': category,
                            'metrics': all_metrics
                        }, os.path.join(self.model_dir, "%d_best_%s.torch" % (config_id, category)))
            metric_history.append(all_metrics)

            print("GPU %d.%d" % (self.gpu_id, self.sub_gpu_id),
                  "| conf", config_id,
                  '| %03d/%03d' % (epoch+1, epochs),
                  '| %ds' % (int(epoch_duration)),
                  '| train %0.6f' % epoch_train_losses[-1],
                  '| val %0.6f' % epoch_val_losses[-1],
                  '| dist %0.6f' % float(dist_loss),
                  '| e68 %0.2f [B %0.2f]' % (metrics["e68"], lowest_error['e68']),
                  '| h68 %0.2f [B %0.2f]' % (metrics["h68"], lowest_error['h68']),
                  '| e49 %0.2f [B %0.2f]' % (metrics["e49"], lowest_error['e49']),
                  '| h49 %0.2f [B %0.2f]' % (metrics["h49"], lowest_error['h49']),
                  )

            if should_stop(val_loss):
                epochs = epoch + 1
                print_info("EarlyStopping (patience = %d, max_ratio=%f) criterion returned true in epoch %d. Stop training" % (should_stop.patience, should_stop.max_ratio, epochs))
                break

        endtime = time.time()

        # Write a loss plot to CONFIG_ID_loss_plot.txt in the output directory
        # TODO tensorboardX in addition to matplotlib?
        x = np.array(range(epochs))
        plt.plot(x, np.array(epoch_train_losses), 'r', label='Train Loss')
        plt.plot(x, np.array(epoch_val_losses), 'b', label='Val Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Avg. Train and Val Loss")
        plt.title("Variation of train and Val loss with epochs")
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.plot_dir, "%d_loss_plot.png" % config_id))
        plt.close()

        training_duration = int(endtime - starttime)

        best_epochs = {"best_%s_epoch" % k: v for k, v in best_epoch.items()}
        best_errors = {"best_%s" % k: v for k, v in lowest_error.items()}

        results = {
            "config_id" : config_id,
            'dataset': self.data,
            "gpu_id" : self.gpu_id,
            "duration_seconds": training_duration,
            "last_epoch" : epochs, # is different from n_epoch in case of early stopping
            "trainable_parameters": trainable_parameters,
            **self.config,
            "optimizer_name" : optimizer.__class__.__name__,
            **best_epochs,
            "training_loss_last_epoch" : epoch_train_losses[-1],
            **best_errors
        }

        # Write results to CONFIG_ID_result.json in the output directory
        with open(os.path.join(self.result_dir, "%d_result.json" % config_id), "w") as f:
            to_write = {
                **results,
                'loss_history' : loss_history,
                'metric_history': metric_history
            }
            json.dump(to_write, f, indent=4)

        torch.cuda.empty_cache()

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_file',
                        type=str,
                        help='Config File')
    parser.add_argument('output_dir',
                        type=str,
                        help='Models, results, plots will be saved here')
    parser.add_argument('data',
                        type=str,
                        help='location of data files to take in')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=None,
                        help='Which GPU to use')

    opt = parser.parse_args()

    model_trainer = ModelTrainer(config=opt.config_file,
                                 output_dir=opt.output_dir,
                                 gpu_id=opt.gpu_id,
                                 sub_gpu_id=0,
                                 gridsearch=False,
                                 data=opt.data)

    results = model_trainer.run()
    for k, v in results.items():
        print(k, "\t->\t", v)
