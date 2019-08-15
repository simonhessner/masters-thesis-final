import cv2
import os
from tqdm import tqdm
import h5py
import numpy as np
from glob import glob
import sys

src = sys.argv[1]
target = sys.argv[2]


def get_lm_coords(fn):
    with open(fn) as f:
        lines = f.read().splitlines()

        if "39" in lines[1]:
            coordlines = lines[3:42]
        else:
            coordlines = lines[3:71]

        assert len(coordlines) in [39, 68]

        return [(float(x[0])-1, float(x[1])-1) for x in [y.split(" ") for y in coordlines]]


with h5py.File(target, "w") as h5:
    tr = h5.create_group("menpo")

    all_images = []
    all_filenames = []
    all_landmarks = []
    n = 0

    for fname in tqdm(glob(os.path.join(src, "*.jpg"))):
        n += 1
        fname = os.path.basename(fname)  # remove path

        # calculate bounding box as the samellest recangle that contains all landmarks
        lms = get_lm_coords(os.path.join(src, fname.replace("jpg", "pts")))
        if len(lms) != 68:
            continue

        x1 = min([c[0] for c in lms])
        x2 = max([c[0] for c in lms])
        y1 = min([c[1] for c in lms])
        y2 = max([c[1] for c in lms])

        img = cv2.imread(os.path.join(src, fname))

        h, w, _ = img.shape
        target_h, target_w = 128, 128
        h_hlf, w_hlf = target_h/2, target_w/2  # needed to transform coordinates into [-1,1]

        # Increase BB size by 25%
        r = 1.05
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

    sources_train = [np.string_("menpo")]*n

    tr.create_dataset("images", data=np.array(all_images))
    tr.create_dataset("coords", data=np.array(all_landmarks, dtype=np.float64))
    tr.create_dataset("filenames", data=np.array(all_filenames), dtype="S%d" % max([len(x) for x in all_filenames]))
    tr.create_dataset("source", data=np.array(sources_train, dtype="S%d" % max([len(x) for x in sources_train])))
