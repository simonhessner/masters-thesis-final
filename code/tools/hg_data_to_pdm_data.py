import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("hg_file")
parser.add_argument("pdm_file")
args = parser.parse_args()

with h5py.File(args.hg_file, "r") as hg, h5py.File(args.pdm_file, "w") as pdm:
    d = pdm.create_group("300W")
    train = d.create_dataset("train_y", data=hg["train/coords"])
    test = d.create_dataset("test_y", data=hg["test/coords"])

