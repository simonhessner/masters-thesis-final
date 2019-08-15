from pe_hourglass.main import ModelTrainer
import torch
import argparse
from torchvision import transforms
from datasets.facial_landmarks import FaceLandmarksTrainingData
from common.transforms import ImageTransform
import cv2
from tools.visualize_dataset import draw_landmarks
import dlib


def run_hg(model, image, bb=False):
    location = torch.device("cpu")

    data = torch.load(model, map_location=location)
    state_dict = data['state_dict']
    config = data['config']

    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(location)

    normMean, normStd = FaceLandmarksTrainingData.TRAIN_MEAN, FaceLandmarksTrainingData.TRAIN_STD
    normTransform = transforms.Normalize(normMean, normStd)

    transform = transforms.Compose([
        ImageTransform(transforms.ToPILImage()),
        ImageTransform(transforms.ToTensor()),
        ImageTransform(normTransform)
    ])

    img = cv2.imread(image)
    imgs = []
    positions = []

    # TODO find a way to detect the whole head
    dnnFaceDetector = dlib.cnn_face_detection_model_v1("../other/mmod_human_face_detector.dat")
    faceRects = dnnFaceDetector(img, 0)

    for rect in faceRects:
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()

        w = x2-x1
        h = y2-y1

        # ensure it is a rectangle
        if h > w:
            diff = h - w
            y2 -= diff
        if w > h:
            diff = w - h
            x2 -= diff

        positions.append((x1,y1,x2,y2))

        face = cv2.resize(img[y1:y2,x1:x2], (128, 128))[:,:,::-1]
        imgs.append(transform({"image" : torch.tensor(face).permute(2,0,1)})["image"])

        if bb:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    imgs = torch.stack(imgs)
    out = net(imgs)[0]

    for coords, (x1,y1,x2,y2) in zip(out.detach().numpy(), positions):
        img[y1:y2,x1:x2] = draw_landmarks(img[y1:y2,x1:x2], coords, size=1)

    cv2.imshow("landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hourglass", type=str, help="Path to hourglass (.torch)")
    parser.add_argument("image", type=str, help="Path to image")
    parser.add_argument("--bb", action='store_true', default=False, help="Show BB for faces")

    args = parser.parse_args()
    run_hg(args.hourglass, args.image, args.bb)