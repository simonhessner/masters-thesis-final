import argparse
import torch
from pe_hourglass.main import ModelTrainer
from common.util import count_parameters
from glob import glob
import os
from tqdm import tqdm
from csv import DictWriter


def count_model_parameters(model):
    location = torch.device("cpu")

    data = torch.load(model, map_location=location)
    state_dict = data['state_dict']
    config = data['config']
    config_id = config["config_id"]

    net = ModelTrainer.create_net(config, verbose=False)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(location)

    _, trainable_parameters, _ = count_parameters(net)
    return config_id, trainable_parameters


def run(sourcedir):
    res = {}
    for model in tqdm(glob(os.path.join(sourcedir, "models", "*_best_hard_noutline.torch"))): # all splits have same amount of params, so it is enough to do only one
        try:
            config_id, trainable_parameters = count_model_parameters(model)
        except RuntimeError as e:
            print("Error loading %s" % model)
            continue

        if config_id in res:
            if res[config_id] != trainable_parameters:
                raise Exception("CONFLICT! new:%d old:%d" % (trainable_parameters, res[config_id]))
        else:
            res[config_id] = trainable_parameters
            tqdm.write("%d \t %d" % (config_id, trainable_parameters))

    reslist = []
    for k,v in res.items():
        reslist.append({"config_id": k, "trainable_parameters": v})

    target = os.path.join(sourcedir, "trainable_parameters.csv")
    with open(target, "w") as f:
        writer = DictWriter(f, ["config_id", "trainable_parameters"])
        writer.writeheader()
        writer.writerows(reslist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sourcedirs", nargs="+")

    args = parser.parse_args()
    for sourcedir in tqdm(args.sourcedirs):
        tqdm.write(sourcedir)
        run(sourcedir)