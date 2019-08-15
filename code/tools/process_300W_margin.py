import argparse
import os
from glob import glob
import h5py
import numpy as np
import cv2


def read_annotations(source):
    with open(source) as f:
        lines = f.read().splitlines()
        n_points = int(lines[1].split("  ")[1])
        coord_lines = lines[3:3+n_points]
        assert len(coord_lines) == n_points
        coords = [(float(x[0])-1, float(x[1])-1) for x in [y.split(" ") for y in coord_lines]]
        return n_points, coords


def process_dataset(source_dir, ending):
    print(source_dir)
    all_coords = []
    filenames = []
    faces = []

    for path in glob(os.path.join(source_dir, "*.%s" % ending)):
        filename = os.path.basename(path)
        filenames.append(filename)

        img = cv2.imread(path)
        image_h, image_w,_ = img.shape

        annotation_source = os.path.join(source_dir, filename.replace(ending, "pts"))
        n_points, coords = read_annotations(annotation_source)
        assert n_points == 68

        # tight BB around landmarks
        xmin = min([c[0] for c in coords])
        xmax = max([c[0] for c in coords])
        ymin = min([c[1] for c in coords])
        ymax = max([c[1] for c in coords])

        # increase BB width and height by factor 1.15
        r = 1.05
        x_middle = (xmin + xmax) / 2
        y_middle = (ymin + ymax) / 2
        bb_width = (xmax - xmin) * r
        bb_height = (ymax - ymin) * r

        # ensure that BB is within image
        x1 = int(max(0, x_middle - bb_width/2))
        x2 = int(min(image_w, x_middle + bb_width/2))
        y1 = int(max(0, y_middle - bb_height/2))
        y2 = int(min(image_h, y_middle + bb_height/2))

        bb_width = x2-x1
        bb_height = y2-y1

        target_width, target_height = 128, 128

        face = img[y1:y2,x1:x2]
        face = cv2.resize(face, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        faces.append(face[:,:,::-1])  # BGR -> RGB

        # the shrinking factor due to resizing
        wf = bb_width/target_width
        hf = bb_height/target_height

        coords = [(int((coord[0]-x1)/wf), int((coord[1]-y1)/hf)) for coord in coords]  # move coordinates to the newly cropped face and correct different scaling due to resize
        coords = [((2*coord[0]/target_width)-1, (2*coord[1]/target_height)-1) for coord in coords]  # convert coordinate into range [-1,1]
        all_coords.append(coords)

    return all_coords, filenames, faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("target_file")
    args = parser.parse_args()

    # Train
    helen_train_path = os.path.join(args.source_dir, "helen", "trainset")
    lfpw_train_path = os.path.join(args.source_dir, "lfpw", "trainset")
    afw_path = os.path.join(args.source_dir, "afw")

    # Test
    helen_test_path = os.path.join(args.source_dir, "helen", "testset")
    lfpw_test_path = os.path.join(args.source_dir, "lfpw", "testset")
    ibug_path = os.path.join(args.source_dir, "ibug")

    with h5py.File(args.target_file, "w") as h5:
        tr = h5.create_group("train")

        train_coords, train_filenames, train_faces, train_sources = [], [], [], []

        # HELEN train
        c,f,i = process_dataset(helen_train_path, "jpg")
        train_coords += c
        train_filenames += f
        train_faces += i
        train_sources += [np.string_("helen/train")]*len(c)

        # LFPW train
        c,f,i = process_dataset(lfpw_train_path, "png")
        train_coords += c
        train_filenames += f
        train_faces += i
        train_sources += [np.string_("lfpw/train")]*len(c)

        # AFW
        c,f,i = process_dataset(afw_path, "jpg")
        train_coords += c
        train_filenames += f
        train_faces += i
        train_sources += [np.string_("afw")]*len(c)

        tr.create_dataset("images", data=np.array(train_faces))
        tr.create_dataset("filenames", data=np.array([np.string_(s) for s in train_filenames]), dtype="S%d" % max([len(x) for x in f]))
        tr.create_dataset("coords", data=np.array(train_coords, dtype=np.float64))
        tr.create_dataset("source", data=np.array(train_sources, dtype="S%d" % max([len(x) for x in train_sources])))

        ### TEST
        te = h5.create_group("test")

        test_coords, test_filenames, test_faces, test_sources = [], [], [], []

        # HELEN test
        c,f,i = process_dataset(helen_test_path, "jpg")
        test_coords += c
        test_filenames += f
        test_faces += i
        train_sources += [np.string_("helen/test")]*len(c)

        # LFPW test
        c,f,i = process_dataset(lfpw_test_path, "png")
        test_coords += c
        test_filenames += f
        test_faces += i
        test_sources += [np.string_("lfpw/test")]*len(c)

        # IBUG
        c,f,i = process_dataset(ibug_path, "jpg")
        test_coords += c
        test_filenames += f
        test_faces += i
        test_sources += [np.string_("ibug")]*len(c)

        te.create_dataset("images", data=np.array(test_faces))
        te.create_dataset("filenames", data=np.array([np.string_(s) for s in test_filenames]), dtype="S%d" % max([len(x) for x in f]))
        te.create_dataset("coords", data=np.array(test_coords, dtype=np.float64))
        te.create_dataset("source", data=np.array(test_sources, dtype="S%d" % max([len(x) for x in test_sources])))