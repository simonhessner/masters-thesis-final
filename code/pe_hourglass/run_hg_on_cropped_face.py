import cv2
import torch
import argparse
import json
from pe_hourglass.main import ModelTrainer
from torchvision import transforms
from datasets.facial_landmarks import FaceLandmarksTrainingData
from common.transforms import ImageTransform
from tools.visualize_dataset import draw_landmarks


def run_hg(model, images, gpu):
    # Load model
    location = "cpu" if gpu is None else "cuda:%d" % gpu
    data = torch.load(model, map_location=location)
    state_dict = data['state_dict']
    config = data['config']
    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(location)

    # Define transformations that normalize the image
    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    # Read images and apply transformations to them
    imgs = []
    #origimg = []
    for img_fn in images:
        img = cv2.imread(img_fn)[:,:,::-1] # BGR -> RGB
        img = cv2.resize(img, (128,128))
        #origimg.append(img)
        imgs.append(transform({"image": torch.tensor(img).permute(2,0,1)})["image"])

    # Run HG on all images (can crash if too many are used)
    imgs = torch.stack(imgs).to(location)
    with torch.no_grad():
        predictions, *_ = net(imgs)

    """
    for i in range(len(origimg)):
        d = draw_landmarks(origimg[i][:,:,::-1], res[i].cpu().detach().numpy())
        cv2.imwrite("/tmp/pred_%d.jpg" %i, d)
    """

    return predictions.detach()


if __name__ == "__main__":
    desc = "Runs a hourglass model on one or multiple images"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=desc)
    parser.add_argument("model", type=str, help="Path to HG h5 file")
    parser.add_argument("images", nargs="+", help="Paths to images that contain cropped faces")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use. Default is CPU")
    args = parser.parse_args()

    result = run_hg(args.model, args.images, args.gpu)
    output = {}
    for img, predictions in zip(args.images, result.cpu().numpy().tolist()):
        output[img] = predictions
    print(json.dumps(output, indent=4))