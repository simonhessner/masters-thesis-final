import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from datasets.facial_landmarks import Menpo, FaceLandmarksTrainingData
from common.transforms import ImageTransform
from torchvision import transforms

from pe_hourglass.main import ModelTrainer
from common.metrics import evaluate_menpo
import h5py
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("menpo")
    args = parser.parse_args()
    model = args.model
    location = "cuda:0"

    data = torch.load(model, map_location=location)
    state_dict = data['state_dict']
    config = data['config']

    num_workers = 8
    batch_size = config['batch_size']
    pin_memory = True

    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()

    net = net.to(location)

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    net.eval()
    with h5py.File(args.menpo, 'r') as f:
        d = Menpo(f, transform=transform)
        d_loader = DataLoader(dataset=d, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)

        gts = []
        preds = []

        with torch.no_grad():
            for batch in tqdm(d_loader):
                gt_lm = batch['landmarks']
                img = batch['image'].to(location)
                predicted_landmarks, heatmaps, var, unnormal_hms = net(img)

                gts.append(gt_lm.detach())
                preds.append(predicted_landmarks.detach())



        menpo_68, menpo_49 = evaluate_menpo(torch.cat(preds).cpu().float(), torch.cat(gts).cpu().float())

        print("menpo 49", menpo_49)
        print("menpo 68", menpo_68)

