import argparse
import h5py
import torch
from datasets.facial_landmarks import FaceLandmarksTrainingData, FaceLandmarksEasyTestData, FaceLandmarksHardTestData, Menpo
from common.transforms import ImageTransform
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from pe_hourglass.main import ModelTrainer
from common.metrics import benchmark, evaluate_menpo, evaluate
import multiprocessing
import numpy as np
import json
import time
from common.util import count_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hg", type=str, help="Path to hourglass model")
    parser.add_argument("dataset_300w", type=str, help="Path to 300W h5 file")
    parser.add_argument("menpo", type=str, help="Path to menpo h5 file")
    parser.add_argument("--gpu", default=None, type=int, help="Run on GPU ID --gpu instead of CPU")
    parser.add_argument("--override_norm_params", default=False, action="store_true")
    args = parser.parse_args()

    gpu = args.gpu
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

    data = torch.load(args.hg, map_location=location)
    state_dict = data['state_dict']
    config = data['config']
    n_lm = config["n_lm"]

    print("Trained for %d epochs" % data["epoch"])
    print()
    print(json.dumps(config, indent=4))

    num_workers = multiprocessing.cpu_count()
    batch_size = config['batch_size'] if gpu is not None else num_workers
    pin_memory = args.gpu is not None

    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()

    _, parameters, _ = count_parameters(net)
    print("Parameters:", parameters)

    net.to(location)

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD

    # TODO remove when this is stored in h5 file
    if args.override_norm_params:
        normMean = tuple(np.array([133.0255852472676, 101.61684197664563, 87.4134193236219]) / 255.0)
        normStd = tuple(np.array([71.91047346327116, 62.94368776888253, 61.56865329427311]) / 255.0)

    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    with h5py.File(args.dataset_300w, 'r') as f:
        easy_d = FaceLandmarksEasyTestData(f, transform=transform, n_lm=n_lm)
        easy_loader = DataLoader(dataset=easy_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)

        hard_d = FaceLandmarksHardTestData(f, transform=transform, n_lm=n_lm)
        hard_loader = DataLoader(dataset=hard_d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)

        print("300W")
        with torch.no_grad():
            net.eval()
            res_300w = benchmark(net, easy_loader, hard_loader, location)
        print("e49", res_300w["e49"])
        print("e68", res_300w["e68"])
        print("h49", res_300w["h49"])
        print("h68", res_300w["h68"])

    print()

    with h5py.File(args.menpo, 'r') as f:
        d = Menpo(f, transform=transform)
        d_loader = DataLoader(dataset=d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)

        gts = []
        preds = []

        durations = []

        net.eval()

        with torch.no_grad():
            for batch in d_loader:
                gt_lm = batch['landmarks']
                img = batch['image'].to(location)

                starttime = time.time()
                predicted_landmarks, heatmaps, var, unnormal_hms = net(img)
                endtime = time.time()
                duration_batch = endtime - starttime
                durations.append(duration_batch)
                #print(duration_batch)

                gts.append(gt_lm.detach())
                preds.append(predicted_landmarks.detach())

        print("mean duration per batch", sum(durations)/len(durations))

        print()
        print("Menpo with BB size error")
        res_menpo = evaluate_menpo(torch.cat(preds).cpu().float(), torch.cat(gts).cpu().float())
        print("Without outline", res_menpo[1])
        print("With outline", res_menpo[0])

        print()
        res_menpo_iod = evaluate(torch.cat(preds).cpu().float(), torch.cat(gts).cpu().float())
        print("menpo with IOD error")
        print("49", res_menpo_iod["without_outline"])
        print("68", res_menpo_iod["with_outline"])


