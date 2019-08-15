import argparse
import h5py
import torch
from datasets.facial_landmarks import FaceLandmarksTrainingData, FaceLandmarksEasyTestData, FaceLandmarksHardTestData, Menpo
from common.util import mkdir_if_not_exists
from common.transforms import ImageTransform
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import os
import multiprocessing
import json
from pe_hourglass.main import ModelTrainer
from common.metrics import evaluate
import numpy as np

def evaluate_split(net, split, location, n_lm):
    results = []

    with torch.no_grad():
        for batch in tqdm(split):
            images = batch['image'].to(location)
            gt = batch['landmarks'].to(location)
            if gt.shape[1] != n_lm:
                raise ValueError("GT has invalid n_lm (%d vs. %d)" % (gt.shape[1], n_lm))
            predicted_landmarks, heatmaps, variance, unnorm_hm = net(images)
            if predicted_landmarks.shape[1] != n_lm:
                raise ValueError("predicted_landmarks has invalid n_lm (%d vs. %d)" % (predicted_landmarks.shape[1], n_lm))
            if variance.shape[1] != n_lm:
                raise ValueError("variance has invalid n_lm (%d vs. %d)" % (variance.shape[1], n_lm))
            predicted_landmarks = predicted_landmarks.detach().cpu().numpy().tolist()
            gt = gt.detach().cpu().numpy().tolist()
            variance = variance.detach().cpu().numpy().tolist()

            for i in range(len(images)):
                coord_and_var = [{
                    "lm_i"   : k,
                    "pred_x" : predicted_landmarks[i][k][0], # predicted x
                    "pred_y" : predicted_landmarks[i][k][1], # predicted y
                    "gt_x"   : gt[i][k][0], # real x
                    "gt_y"   : gt[i][k][1], # real y,
                    "var_x"  : variance[i][k][0], # var x
                    "var_y"  : variance[i][k][1], # var y
                } for k in range(n_lm)]

                results.append({
                    "coord_and_variance"  : coord_and_var,
                    #"heatmaps_norm"   : heatmaps[i].detach().cpu().numpy().tolist(),
                    #"heatmaps_unnorm" : unnorm_hm[i].detach().cpu().numpy().tolist()
                })

        results = {
            "results": results
        }

    return results


def run(model, src_300w, src_menpo, target, gpu=None, override_norm_params=False, bs_factor=1):
    location = 'cpu' if gpu is None else "cuda:%d" % gpu
    if location is not 'cpu':
        # This fixes the problem that pytorch is always allocating memory on GPU 0 even if this is not included
        # in the list of GPUs to use
        torch.cuda.set_device(torch.device(location))

        # cudnn.benchmark improves training speed when input sizes do not change
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        # It selects the best algorithms as the training iterates over the dataset
        #cudnn.benchmark = True # disable for deterministic behavior

    print("Location: ", location)

    data = torch.load(model, map_location=location)
    state_dict = data['state_dict']
    config = data['config']
    n_lm = config["n_lm"]

    if n_lm == 49:
        print("WARNING! THIS IS A 49 LM model!!!!", n_lm)

    num_workers = multiprocessing.cpu_count()
    batch_size = config['batch_size'] * bs_factor if gpu is not None else num_workers
    pin_memory = gpu is not None

    print("Workers: ", num_workers)
    print("Batchsize: ", batch_size)

    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()

    net.to(location)

    mkdir_if_not_exists(os.path.dirname(target))

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD

    if override_norm_params:
        normMean = tuple(np.array([133.0255852472676, 101.61684197664563, 87.4134193236219]) / 255.0)
        normStd = tuple(np.array([71.91047346327116, 62.94368776888253, 61.56865329427311]) / 255.0)

    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    with h5py.File(src_300w, 'r') as f:
        print("Run on easy")
        easy_d = FaceLandmarksEasyTestData(f, transform=transform, n_lm=n_lm)
        easy_loader = DataLoader(dataset=easy_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
        easy_results = evaluate_split(net, easy_loader, location=location, n_lm=n_lm)

        print("Run on hard")
        hard_d = FaceLandmarksHardTestData(f, transform=transform, n_lm=n_lm)
        hard_loader = DataLoader(dataset=hard_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
        hard_results = evaluate_split(net, hard_loader, location=location, n_lm=n_lm)

        print("Run on train")
        train = FaceLandmarksTrainingData(f, transform=transform, n_lm=n_lm)
        train_loader = DataLoader(dataset=train, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
        train_results = evaluate_split(net, train_loader, location=location, n_lm=n_lm)

    with h5py.File(src_menpo, "r") as f:
        print("Run on menpo")
        menpo = Menpo(f, transform=transform, n_lm=n_lm)
        menpo_loader = DataLoader(dataset=menpo, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
        menpo_results = evaluate_split(net, menpo_loader, location=location, n_lm=n_lm)

    res = {
        "easy": easy_results,
        "hard": hard_results,
        "train": train_results,
        "menpo": menpo_results,
        "model_src": model,
        "config": config
    }

    if target is not None:
        json.dump(res, open(target, "w"))
    else:
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hg", type=str, help="Path to hourglass model")
    parser.add_argument("src_300w", type=str, help="Path to 300W h5 file")
    parser.add_argument("src_menpo", type=str, help="Path to menpo h5 file")
    parser.add_argument("target", type=str, help="Target file (.json)")
    parser.add_argument("--gpu", default=None, type=int, help="Run on GPU ID --gpu instead of CPU")
    parser.add_argument("--override_norm_params", default=False, action="store_true")
    parser.add_argument("--bs_factor", type=int, default=1)
    args = parser.parse_args()

    run(model=args.hg, src_300w=args.src_300w, src_menpo=args.src_menpo, target=args.target, gpu=args.gpu, override_norm_params=args.override_norm_params, bs_factor=args.bs_factor)