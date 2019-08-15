import sys
import torch
import h5py
import pickle
from pdm.main import ModelTrainer

location = "cpu"

data = torch.load(sys.argv[1], map_location=location)
state_dict = data['state_dict']
config = data['config']

pdm = ModelTrainer.create_net(config)
pdm.model.load_state_dict(state_dict)

data = "/home/simon/Desktop/InterACT/Masterarbeit/Code/cvpr18-combined-model/pdm/all_data_valid_w_profile_pts.h5"
with h5py.File(data) as data:
    zs, nr, loss = pdm.test(torch.tensor(data["test_y"], dtype=torch.float32))
    l2d, l3d = pdm.forward(zs, nr)

    l2d = l2d.detach()
    l3d = l3d.detach()

    pickle.dump({
        "l2d": l2d.cpu().numpy(),
        "before_pdm": None,
        "gt": None,
        "l3d": l3d.cpu().numpy()
    }, open("/tmp/test_debugging_pdm_results.pickle", "wb"))