import cv2
import scipy.io as sio
import os
#import math
from tqdm import tqdm
import h5py
import numpy as np

src = "/home/simon/Desktop/AFW"
bbsrc = os.path.join(src, "bounding_boxes_afw.mat")


def get_lm_coords(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
        coordlines = lines[3:71]

        return [(float(x[0])-1, float(x[1])-1) for x in [y.split(" ") for y in coordlines]]


bbs = sio.loadmat(bbsrc)["bounding_boxes"][0]

with h5py.File("/tmp/hourglass_samples_with_afw_without_multipie.h5", "w") as h5, h5py.File("/home/simon/Desktop/InterACT/Masterarbeit/Code/facial_landmarks_from_holmes_ceclm_68_split.h5") as oldfile:
    te = h5.create_group("test")
    te.create_dataset("filenames", data=oldfile["test/filenames"])
    te.create_dataset("coords", data=oldfile["test/coords"])
    te.create_dataset("images", data=oldfile["test/images"])

    sources_test = [np.string_("helen/test")]*330 + [np.string_("lfpw/test")]*224 + [np.string_("ibug")]*135
    te.create_dataset("source", data=np.array(sources_test), dtype="S%d" % max([len(x) for x in sources_test]))

    tr = h5.create_group("train")
    all_landmarks = list(oldfile["train/coords"])[0:2811]
    all_images = list(oldfile["train/images"])[0:2811]
    all_filenames = list(oldfile["train/filenames"])[0:2811]

    for i in tqdm(range(len(bbs))):
        fname = bbs[i][0][0][0][0]

        # calculate bounding box as the samellest recangle that contains all landmarks
        lms = get_lm_coords(os.path.join(src, fname.replace("jpg", "pts")))
        x1 = min([c[0] for c in lms])
        x2 = max([c[0] for c in lms])
        y1 = min([c[1] for c in lms])
        y2 = max([c[1] for c in lms])

        #bb1 = [x-1 for x in bbs[i][0][0][1][0]]
        #bb = [x-1 for x in bbs[i][0][0][2][0]]  # bigger
        # coordinates defining the bounding box
        #x1, x2, y1, y2 = bb[0], bb[2], bb[1], bb[3]

        img = cv2.imread(os.path.join(src, fname))

        h, w, _ = img.shape
        target_h, target_w = 128, 128
        h_hlf, w_hlf = target_h/2, target_w/2  # needed to transform coordinates into [-1,1]

        # Increase BB size by 25%
        r = 1.25
        mid_x = (x1+x2)/2
        mid_y = (y1+y2)/2
        width = x2-x1
        height = y2-y1
        new_width = width * r
        new_height = height * r
        x1 = int(max(0, mid_x - (new_width / 2)))
        x2 = int(min(w, mid_x + (new_width / 2)))
        y1 = int(max(0, mid_y - (new_height / 2)))
        y2 = int(min(h, mid_y + (new_height / 2)))

        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (target_w, target_h))
        face = face[:,:,::-1]  # RGB

        face_width = x2-x1
        face_height = y2-y1
        wf = face_width/target_w  # the shrinking factor due to resizing
        hf = face_height/target_h

        lms = [(int((coord[0]-x1)/wf), int((coord[1]-y1)/hf)) for coord in lms]  # move coordinates to the newly cropped face and correct different scaling due to resize
        lms = [((coord[0]/w_hlf)-1, (coord[1]/h_hlf)-1) for coord in lms]  # convert coordinate into range [-1,1]

        all_images.append(face)
        all_landmarks.append(lms)
        all_filenames.append(np.string_(os.path.splitext(fname)[0]))

    sources_train = [np.string_("helen/train")]*2000 + [np.string_("lfpw/train")]*811 + [np.string_("afw")]*len(bbs)

    tr.create_dataset("images", data=np.array(all_images))
    tr.create_dataset("coords", data=np.array(all_landmarks, dtype=np.float64))
    tr.create_dataset("filenames", data=np.array(all_filenames), dtype="S%d" % max([len(x) for x in all_filenames]))
    tr.create_dataset("source", data=np.array(sources_train, dtype="S%d" % max([len(x) for x in sources_train])))
