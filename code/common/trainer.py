import torch
from common.util import move2device

class BaseModelTrainer(object):
    def __init__(self, gpu_id, sub_gpu_id=0):
        self.gpu_id = gpu_id
        self.sub_gpu_id = sub_gpu_id
        # This fixes the problem that pytorch is always allocating memory on GPU 0 even if this is not included
        # in the list of GPUs to use
        if self.gpu_id not in [None, -1]:
            torch.cuda.set_device(torch.device("cuda:%d" % gpu_id))

    @staticmethod
    def create_net(config):
        raise NotImplementedError

    def to_gpu(self, tensor):
        return move2device(tensor, self.gpu_id)

    def run(self):
        raise NotImplementedError

    def get_device(self):
        if self.gpu_id in [None, -1]:
            return torch.device("cpu")
        return torch.device("cuda:%d" % self.gpu_id)
