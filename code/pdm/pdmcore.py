import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from common.metrics import WingLoss, evaluate
from common.util import extract_key
import json
from torch.nn.modules.loss import _Loss
from common.metrics import NO_OUTLINE_MASK

# to avoid uncommenting @profile all the time
try:
  dummy = profile
except:
  profile = lambda x: x


class PDM():
    ALPHA_HG_SOURCE_49 = "../../models/hg_alpha/49"  # must contain one json file per split
    ALPHA_HG_SOURCE_68 = "../../models/hg_alpha/68"  # must contain one json file per split

    def __init__(self,
                 layers,
                 train_epochs,
                 test_epochs,
                 bs,
                 lr_shape,
                 lr_net,
                 is_68_lm_pdm=True,
                 activation="relu",
                 last_linear=True,
                 multiplicative_confidence=False,
                 net_scheduler=None,
                 batch_norm=False, # is ignored, keep for compatibility
                 conf_params=(1.0, 0.0),
                 loss=("wing", 10.0, 1.0),
                 noise_std=0.0,
                 shape_init=("uniform", 0.1),
                 shuffle=True,
                 target=None): # for compatibility
        """
        Encapsulates a point distribution model (PDM) and provides methods to train and infer 2D/3D shapes from the model

        :param layers: List of number of neurons per layer. First layer is hidden vector size.
        :param last_linear: If there is only one layer and this is true, the whole model is linear
        :param train_epochs: Number of epochs during training
        :param test_epochs: Number of epochs during test
        :param bs: Batch size
        :param lr_net: learning rate for shape parameters (zs, nr)
        :param lr_shape: learning rate for network weights
        """
        n_lm = 68 if is_68_lm_pdm else 49
        _layers = [nn.Linear(layers[i - 1], layers[i]) for i in range(1, len(layers))] + [nn.Linear(layers[-1], n_lm * 3)]
        prototype = []
        for i,l in enumerate(_layers):
            prototype.append(l)
            if i < len(_layers) - 1 or not last_linear:
                if activation == "relu":
                    prototype.append(nn.ReLU(inplace=False))
                elif activation == "sigmoid":
                    prototype.append(nn.Sigmoid())
                elif activation == "tanh":
                    prototype.append(nn.Tanh())
                else:
                    raise ValueError("Unknown activation function %s" % activation)
        self.model = nn.Sequential(*prototype)
        self.zdim = layers[0]

        self.device = 'cpu'

        self.print_losses = True
        self.verbose = True
        self.listener = None

        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.bs = bs
        self.lr_shape = lr_shape
        self.lr_net = lr_net
        self.multiplicative_confidence = multiplicative_confidence
        self.net_scheduler = net_scheduler

        self.test_scheduler = None
        self.test_optimizer = {"name": "adam", "lr": self.lr_shape}
        self.test_reduction = "sum"  # This is just to be able to reproduce the old results by setting to "mean"

        self.is_68_pdm = is_68_lm_pdm
        self.n_lm = 68 if self.is_68_pdm else 49
        self.eval_frequency = 1000
        self.alpha_hg_data = None

        self.conf_a, self.conf_b = conf_params
        self.loss = loss

        self.noise_std = noise_std
        self.shuffle = shuffle

        self.shape_init_method, self.shape_init_std = shape_init
        if self.shape_init_method not in ["uniform", "gaussian"]:
            raise ValueError("Invalid shape_init_method %s" % self.shape_init_method)

    def load_hg_results(self, src, split):
        with open(src, "r") as f:
            data = json.load(f)
            n_hg = data["config"]["n_lm"]

            if self.verbose:
                print("[%s] load %s [%s]" % (split, src, data["model_src"]))

            n_pdm = self.n_lm
            if n_hg != n_pdm:
                raise ValueError("Incompatible HG (%d) and PDM (%d) n_lm" % (n_hg, n_pdm))

            device = torch.device("cpu")  # self.device # TODO GPU instead?
            coord_var = [x["coord_and_variance"] for x in data[split]["results"]]
            gt = torch.tensor([[[y["gt_x"], y["gt_y"]] for y in x] for x in coord_var], device=device)
            hg_pred = torch.tensor([[[y["pred_x"], y["pred_y"]] for y in x] for x in coord_var], device=device)
            hg_conf = torch.tensor([[[self.variance2confidence(y["var_x"]), self.variance2confidence(y["var_y"])] for y in x] for x in coord_var], device=device)

            if gt.shape[1] != n_pdm:
                raise ValueError("gt.shape (%d) != n_pdm (%d)" % (gt.shape[1], n_pdm))

            if hg_pred.shape[1] != n_pdm:
                raise ValueError("hg_pred.shape (%d) != n_pdm (%d)" % (hg_pred.shape[1], n_pdm))

            if hg_conf.shape[1] != n_pdm:
                raise ValueError("hg_conf.shape (%d) != n_pdm (%d)" % (hg_conf.shape[1], n_pdm))

            return {
                "gt" : gt,
                "hg_pred" : hg_pred,
                "hg_conf" : hg_conf
            }

    def init_hg_results(self):
        if self.alpha_hg_data is None:
            self.alpha_hg_data = {}
            srcdir = self.ALPHA_HG_SOURCE_68 if self.is_68_pdm else self.ALPHA_HG_SOURCE_49
            if self.is_68_pdm:
                self.alpha_hg_data["68_lm"] = {
                    "easy": self.load_hg_results(os.path.join(srcdir, "e68.json"), "easy"),
                    "hard": self.load_hg_results(os.path.join(srcdir, "h68.json"), "hard")
                }
            self.alpha_hg_data["49_lm"] = {
                "easy": self.load_hg_results(os.path.join(srcdir, "e49.json"), "easy"),
                "hard": self.load_hg_results(os.path.join(srcdir, "h49.json"), "hard")
            }

    def to(self, device):
        """
        Moves the encapsulated pytorch model to the given device.

        :param device: where to move the model
        :return: model on device
        """
        self.device = device
        self.model  = self.model.to(self.device)
        return self

    def _get_wrapped_loss(self, loss):
        """
        Wraps _Loss subclasses and returns both the batch loss and the average loss

        :param loss: loss to wrap
        :return: batch_loss, sample_loss
        """
        if not isinstance(loss, _Loss):
            raise ValueError("Loss must inherit from _Loss")

        if loss.reduction != 'none':
            raise ValueError("In order to wrap the loss, reduction must be 'none'")

        def wrapped_loss(prediction, gt):
            batchsize = prediction.shape[0]
            sample_loss = loss(prediction, gt).view(batchsize, -1).mean(dim=1)
            batch_loss = sample_loss.mean()
            return batch_loss, sample_loss

        return wrapped_loss

    def get_loss_function(self, return_sample_loss=False, reduction="mean"):
        """
        Can be used to wrap functions like MSELoss or L1Loss so that they return both the loss per sample and also the
        average loss of the batch

        :param return_sample_loss: True => return batch_loss, sample_loss / False => return batch_loss
        :return: batch_loss [, sample_loss]
        """
        loss_type = self.loss[0]
        if loss_type == "wing":
            wing_width = self.loss[1]
            wing_curvature = self.loss[2]
            return WingLoss(width=wing_width, curvature=wing_curvature, return_sample_loss=return_sample_loss, reduction=reduction)

        if loss_type == "l1":
            if return_sample_loss:
                return self._get_wrapped_loss(nn.L1Loss(reduction='none'))
            return nn.L1Loss(reduction=reduction)

        if loss_type == "mse":
            if return_sample_loss:
                return self._get_wrapped_loss(nn.MSELoss(reduction='none'))
            return nn.MSELoss(reduction=reduction)

        raise Exception("Unknown loss function %s" % loss_type)

    def build_mask(self, data):
        """
        For each coodinate (x,y) in the data, a mask is created. If x/y is >= -9.0, then the mask is 1, otherwise 0.
        The output of the model is multiplied with this mask before calculating the loss. This means that if a landmark
        is occluded, no loss for this landmark is computed.

        In the future we will have no binary mask but a mask that can tell the model how confident the previous model
        (e.g. densenet / hourglass) was. It will then only optimize points that have a confidence < 1 and the lower
        the confidence is, the more these points will be optimized.


        :param data: coodinates (for each LM x and y coord)
        :return: for each landmark (0,0) if it is occluded and (1,1) if not.
        """
        if data.shape[2] != 2:
            raise ValueError("data must have shape x,y,2 and not %d in the last dimension" % data.shape[2])

        return torch.ge(data, -9.0).requires_grad_(False).type(dtype=torch.float32).to(data.device)

    def initialize_shape_parameters(self, data):
        """
        Initializes both the latent vector and the affine parameters for each sample randomly

        :param data: N x/y coordinates of landmarks
        :return: zs (hidden vector), nr (affine parameters) [for each sample]
        """
        #zs = torch.empty((data.shape[0], self.zdim), device=data.device)
        if self.shape_init_method == "uniform":
            np_zs = np.random.uniform(low=-self.shape_init_std, high=self.shape_init_std, size=(data.shape[0], self.zdim))
        elif self.shape_init_method == "gaussian":
            np_zs = np.random.normal(loc=0.0, scale=self.shape_init_std, size=(data.shape[0], self.zdim))
        else:
            raise ValueError("Invalid shape_init_method %s" % self.shape_init_method)
        zs = torch.tensor(np_zs, device=data.device).float().requires_grad_(True)

        # rx,ry,rz,s_x,s_y_s_z,t_x,_ty_tz (rotation and translation = 0, scale = 1)
        affine_init = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        nr = torch.tensor(affine_init, dtype=torch.float32, device=data.device).repeat(data.shape[0], 1).requires_grad_(True)

        return zs, nr

    def get_perturbed(self, zs):
        if self.noise_std <= 0.0:
            return zs
        # using numpy gives more stable results when changing the batch sizes
        # so do not use torch.normal_() but numpy here!
        perturbation_np = np.random.normal(loc=0.0, scale=self.noise_std, size=zs.shape)
        perturbation = torch.tensor(perturbation_np, device=zs.device).float()
        perturbed = zs + perturbation
        return perturbed

    @staticmethod
    def transform(in_tensor, nr):
        """
        Given affine parameters, a batch of tensors is transformed by performing scale, rotation and translation in 3D

        :param in_tensor: batch of tensors containing 3D coordinates, BS x LM x 3
        :param nr: 9 affine parameters (scale x/y/z, rotation x/y/z, translation x/y/z)
        :return: input tensor after applying transformation
        """
        batchsize = in_tensor.shape[0]
        dev = in_tensor.device
        dt = torch.float32

        zeros = torch.zeros((batchsize,), device=dev, dtype=dt)
        ones = torch.ones((batchsize,), device=dev, dtype=dt)

        # build scale matrix
        scale_x = nr[:, 3]
        scale_y = nr[:, 4]
        scale_z = nr[:, 5]
        S = torch.stack([
            scale_x, zeros, zeros,
            zeros, scale_y, zeros,
            zeros, zeros, scale_z
        ]).permute(1, 0).view(-1, 3, 3)

        # build rotation matrix for X
        cos_x = torch.cos(nr[:, 0])
        sin_x = torch.sin(nr[:, 0])
        R_X = torch.stack([
            ones, zeros, zeros,
            zeros, cos_x, -sin_x,
            zeros, sin_x, cos_x
        ]).permute(1, 0).view(-1, 3, 3)

        # build scale and rotation matrix for Y
        cos_y = torch.cos(nr[:, 1])
        sin_y = torch.sin(nr[:, 1])
        R_Y = torch.stack([
            cos_y, zeros, sin_y,
            zeros, ones, zeros,
            -sin_y, zeros, cos_y
        ]).permute(1, 0).view(-1, 3, 3)

        # build scale and rotation matrix for Z
        cos_z = torch.cos(nr[:, 2])
        sin_z = torch.sin(nr[:, 2])
        R_Z = torch.stack([
            cos_z, -sin_z, zeros,
            sin_z, cos_z, zeros,
            zeros, zeros, ones
        ]).permute(1, 0).view(-1, 3, 3)

        transformation = R_Z @ (R_Y @ (R_X @ S))  # BS x 3 x 3

        inT = in_tensor.permute(0, 2, 1)  # BS x 3 x LM
        temp = torch.bmm(transformation, inT)  # BSx3x3 * BSx3xLM -> BSx3xLM
        temp = temp.permute(0, 2, 1)  # BS x LM x 3

        T = nr[:, 6:9]  # BS x 3
        temp = temp.permute(1, 0, 2)  # LM x BS x 3
        temp += T
        temp = temp.permute(1, 0, 2)  # BS x LM x 3

        return temp

    def forward(self, zs, nr):
        """
        Given the hidden vector and the affine parameters, a 3D shape is generated using the PDM. This 3D shape is then
        transformed using the affine parameters and projected to a 2D shape by dropping the z-coordinate.

        :param zs: Hidden vector (input for PDM)
        :param nr: Affine parameters
        :return: 2D landmarks, 3D landmarks
        """
        landmarks_3d_flat = self.model(zs)
        landmarks_3d = landmarks_3d_flat.view(landmarks_3d_flat.shape[0], -1, 3)
        landmarks_2d = PDM.transform(landmarks_3d, nr)[:, :, :2]  # project from 3D to 2D by ignoring z-component
        return landmarks_2d, landmarks_3d

    def variance2confidence(self, variance):
        confidence = 1 / (self.conf_a * variance + self.conf_b)
        return confidence

    def eval_on_alpha_hg(self):
        self.init_hg_results()
        eval_bs = 2048

        # do 49LM easy
        d = self.alpha_hg_data["49_lm"]["easy"]
        gt, hg_pred, hg_conf = d["gt"], d["hg_pred"], d["hg_conf"]
        zs, nr, easy_loss, zs_best, nr_best, best_epochs_easy, sample_losses, _, _ = self.test(hg_pred.to(self.device, copy=True), hg_conf.to(self.device, copy=True), verbose=False, bs=eval_bs)
        l2d, _ = self.forward(zs, nr)
        easy49_err = evaluate(l2d, gt.to(self.device, copy=True))["without_outline"]
        torch.cuda.empty_cache()

        # do 49LM hard
        d = self.alpha_hg_data["49_lm"]["hard"]
        gt, hg_pred, hg_conf = d["gt"], d["hg_pred"], d["hg_conf"]
        zs, nr, easy_loss, zs_best, nr_best, best_epochs_easy, sample_losses, _, _ = self.test(hg_pred.to(self.device, copy=True), hg_conf.to(self.device, copy=True), verbose=False, bs=eval_bs)
        l2d, _ = self.forward(zs, nr)
        hard49_err = evaluate(l2d, gt.to(self.device, copy=True))["without_outline"]
        torch.cuda.empty_cache()

        if self.is_68_pdm:
            # do 68LM easy
            d = self.alpha_hg_data["68_lm"]["easy"]
            gt, hg_pred, hg_conf = d["gt"], d["hg_pred"], d["hg_conf"]
            zs, nr, easy_loss, zs_best, nr_best, best_epochs_easy, sample_losses, _, _ = self.test(hg_pred.to(self.device, copy=True), hg_conf.to(self.device, copy=True), verbose=False, bs=eval_bs)
            l2d, _ = self.forward(zs, nr)
            easy68_err = evaluate(l2d, gt.to(self.device, copy=True))["with_outline"]
            torch.cuda.empty_cache()

            # do 68LM hard
            d = self.alpha_hg_data["68_lm"]["hard"]
            gt, hg_pred, hg_conf = d["gt"], d["hg_pred"], d["hg_conf"]
            zs, nr, easy_loss, zs_best, nr_best, best_epochs_easy, sample_losses, _, _ = self.test(hg_pred.to(self.device, copy=True), hg_conf.to(self.device, copy=True), verbose=False, bs=eval_bs)
            l2d, _ = self.forward(zs, nr)
            hard68_err = evaluate(l2d, gt.to(self.device, copy=True))["with_outline"]
            torch.cuda.empty_cache()
        else:
            hard68_err = 100000.0
            easy68_err = 100000.0

        return {
            "e49": easy49_err,
            "h49": hard49_err,
            "e68": easy68_err,
            "h68": hard68_err
        }

    def save_pdm(self, epoch, target):
        torch.save({
            'model': 'pdm',
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'val_loss': 0.0,
            'config': self.config,
        }, target)

    @profile
    def train(self, data):
        """
        Runs the PDM in train mode. In this model both the shape parameters (zs = hidden, nr = affine) and the network
        weights are learned so that they minimize the reconstruction loss.

        :param data: 2D points that should be generated by the hidden vectors and the affine params learned
        :param validation_data: Dataset used to compute validation loss
        :return: zs, nr
        """

        self.model.train()

        if not self.is_68_pdm:
            data = data[:, NO_OUTLINE_MASK, :]

        # TODO if data augmentation enabled, duplicate data and add random scale and translation
        # The data augmentation must be done before training and not be changed from epoch to epoch
        # because the data augmentation directly affects nr (which can not converge if the landmarks move each epoch)
        # To speed things up, maybe only augment a subset (or multiple subsets differently)
        # Force the zs to be the same??

        lossfct = self.get_loss_function(reduction="sum")  # sum for numerical stability
        zs, nr = self.initialize_shape_parameters(data)
        num_data = data.shape[0]
        if data.shape[2] != 2:
            raise ValueError("dimension 2 of data must be 2 in train()")
        mask = self.build_mask(data)
        mask = mask.detach().requires_grad_(False)
        data = data.detach().requires_grad_(False)

        # learn model weights only during training
        adam_main_net = optim.Adam(self.model.parameters(), lr=self.lr_net)
        if self.net_scheduler:
            stepsize = int(self.net_scheduler[0])
            factor = float(self.net_scheduler[1])
            scheduler_main_net = StepLR(optimizer=adam_main_net, step_size=stepsize, gamma=factor)

        # learn hidden vector and affine parameters in both training and test
        # One Adam for each sample because otherwise when using batches, there are updates for batches
        # due to momentum everytime another batch is trained. Also this is needed to be able to use a LR scheduler
        shape_opts = [optim.Adam(params=[zs[i].detach().requires_grad_(True), nr[i].detach().requires_grad_(True)],
                                 lr=self.lr_shape) for i in range(num_data)]  # TODO add scheduler (??? maybe coupled with main net sched)

        epochs = self.train_epochs

        if self.verbose:
            iter = tqdm(range(epochs+1), desc="Mode: train")
        else:
            iter = range(epochs+1)

        for epoch in iter:
            if self.shuffle:
                perm = torch.randperm(num_data, device=data.device)
            else:
                perm = torch.tensor(list(range(num_data)), device=data.device)

            losses = []
            for j in range(num_data // self.bs + 1):
                index_s, index_e = j * self.bs, min((j + 1) * self.bs, num_data)
                if index_e <= index_s:
                    continue
                batch_indices = perm[index_s:index_e]
                batch_indices_list = batch_indices.detach().cpu().numpy().tolist()

                l2d, l3d = self.forward(zs[batch_indices], nr[batch_indices])

                batch_mask = mask[batch_indices]
                if self.multiplicative_confidence:
                    batch_mask = batch_mask * torch.empty(batch_mask.shape, device=batch_mask.device).uniform_(0.0, 1.0)
                loss = lossfct(l2d * batch_mask, data[batch_indices] * batch_mask)

                # The gradient of the whole dataset must be reset because loss.backward() will accumulate in these
                # variables. Calling zero_grad on the sample optimizers will only clear the local gradient of the
                # slice that belongs to that sample and not the global gradients of zs,nr
                zs.grad = None
                nr.grad = None

                adam_main_net.zero_grad()
                loss.backward()
                adam_main_net.step()

                # Copy gradients from zs and nr to the slices, then the optimizer has the gradient it needs and can make
                # a step
                for i_sched in batch_indices_list:
                    shape_opts[i_sched].param_groups[0]["params"][0].grad = zs.grad[i_sched]
                    shape_opts[i_sched].param_groups[0]["params"][1].grad = nr.grad[i_sched]
                    shape_opts[i_sched].step()
                    # The following is not crucial (since we override it later by coping the gradients from zs, nr),
                    # but just to be sure that no gradient accumulates,
                    # set the gradient to None. Calling shape_opt.zero_grad does not work:
                    # RuntimeError: Can't detach views in-place. Use detach() instead
                    shape_opts[i_sched].param_groups[0]["params"][0].grad = None
                    shape_opts[i_sched].param_groups[0]["params"][1].grad = None

                if self.noise_std > 0:
                    zs.data[batch_indices] = self.get_perturbed(zs[batch_indices])

                losses.append(loss.detach().cpu().numpy())

            metrics = None
            if (epoch % self.eval_frequency == 0 and epoch > 0) or (epoch == epochs-1):
                metrics = self.eval_on_alpha_hg()

            epoch_loss = np.mean(np.array(losses))
            if self.verbose:
                tqdm.write("Epoch %03d - Train loss %f" % (epoch + 1, epoch_loss))
                if metrics is not None:
                    print(metrics)
            else:
                if self.listener is not None:
                    self.listener({
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "mode": "train",
                        "metrics" : metrics,
                        "pdm" : self
                    })

            if self.net_scheduler:
                scheduler_main_net.step()

        return zs, nr, epoch_loss  # this will be the last epoch's loss

    @profile
    def test(self, data, confidence, return_history=False, verbose=True, encoder=None, only_encoder=False, bs=None):
        """
        Runs the PDM in test mode. In this mode, only the shape parameters (zs = hidden, nr = affine) are learned.
        The goal is to find a hidden vector and a transformation that can minimize the reconstruction loss. Since
        the network has only seen valid faces during training and knows how to infer zs and nr from any given 2D landmarks,
        it will find zs,nr for any set of landmarks that will represent a valid face that is most similar to the given
        face, but which will have corrected mis-located landmarks (hopefully).

        The loss is masked with the hourglasses confidence.
        That way the model will only backpropagate through the points with a confidence <1 and the lower the confidence,
        the more likely it gets that the model will move this landmark.

        :param data: 2D points that should be reproduces by the latent vectors and the affine params learned
        :return: zs, nr
        """
        self.model.eval()

        n_lm = 68
        if not self.is_68_pdm: # This is the 49 LM case
            n_lm = 49
            if data.shape[1] == 68:
                data = data[:, NO_OUTLINE_MASK, :]
            if confidence is not None and confidence.shape[1] == 68:
                confidence = confidence[:, NO_OUTLINE_MASK, :]

        #print(confidence.shape, data.shape, self.is_68_pdm, n_lm)

        red = self.test_reduction  #Should be "sum" for better numerical stability when using batches
        lossfct = self.get_loss_function(return_sample_loss=True, reduction=red)
        num_data = data.shape[0]
        data = data.detach().requires_grad_(False)
        if confidence is None:
            mask = self.build_mask(data).detach().requires_grad_(False)
        else:
            mask = confidence.detach().requires_grad_(False)

        zs_encoder = None
        nr_encoder = None
        if encoder is None:
            zs, nr = self.initialize_shape_parameters(data)
            if only_encoder:
                return zs, nr, None, None
        else:
            _x = encoder(data.contiguous().view(-1, 2 * n_lm))
            zs, nr = encoder.get_separated(_x)
            zs = zs.detach().requires_grad_(True)
            nr = nr.detach().requires_grad_(True)

            if only_encoder:
                return zs, nr, None, None

            # Remember encoder output and return it in the end (used in NIPS experiments)
            zs_encoder = zs.detach().clone()
            nr_encoder = nr.detach().clone()

        epochs = self.test_epochs

        use_scheduler = self.test_scheduler is not None

        optim_name, optim_params = extract_key(self.test_optimizer, "name")
        if optim_name == "adam":
            # One Adam for each sample because otherwise when using batches, there are updates for batches
            # due to momentum everytime another batch is trained. Also this is needed to be able to use a LR scheduler
            shape_opts = [optim.Adam(params=[zs[i].detach().requires_grad_(True),
                                             nr[i].detach().requires_grad_(True)],
                                     **optim_params) for i in range(num_data)]
        else:
            raise ValueError("Unknown optimizer required: %s" % optim_name)

        if use_scheduler:
            scheduler_name, scheduler_params = extract_key(self.test_scheduler, "name")
            if scheduler_name == "plateau":
                schedulers = [optim.lr_scheduler.ReduceLROnPlateau(shape_opt, **scheduler_params)
                              for shape_opt in shape_opts]  # pat=100, factor=0.1 works
            else:
                raise ValueError("Unknown scheduler required: %s" % scheduler_name)

        if self.verbose and verbose:
            iter = tqdm(range(epochs), desc="Mode: test")
        else:
            iter = range(epochs)

        history = []
        lowest_loss = np.array([10000.0] * num_data)
        best_zs = torch.zeros(zs.shape).cpu().numpy()
        best_nr = torch.zeros(nr.shape).cpu().numpy()
        best_epoch = torch.zeros(best_zs.shape[0]).cpu().numpy()

        sample_losses = []
        loss_history = []

        if bs is None:
            bs = self.bs

        for epoch in iter:
            losses = []
            epoch_results = []

            for j in range(num_data // bs + 1):
                index_s, index_e = j * bs, min((j+1)*bs, num_data)
                if index_e <= index_s:
                    continue

                l2d, l3d = self.forward(zs[index_s:index_e], nr[index_s:index_e])
                epoch_results.extend(l2d.detach().cpu().numpy().tolist())

                batch_mask = mask[index_s:index_e]
                #print(l2d.shape, batch_mask.shape, data.shape)
                loss, sample_loss = lossfct(l2d * batch_mask, data[index_s:index_e] * batch_mask)

                # determine best zs, nr, epoch number per epoch
                batch_lowest_losses = lowest_loss[index_s:index_e]
                sample_loss_np = sample_loss.detach().cpu().numpy()
                sample_losses.extend(sample_loss_np.tolist())
                cmp = sample_loss_np < batch_lowest_losses
                batch_lowest_losses[cmp] = sample_loss_np[cmp]
                lowest_loss[index_s:index_e] = batch_lowest_losses
                best_zs[index_s:index_e][cmp] = zs[index_s:index_e].detach().cpu().numpy()[cmp]
                best_nr[index_s:index_e][cmp] = nr[index_s:index_e].detach().cpu().numpy()[cmp]
                best_epoch[index_s:index_e][cmp] = epoch

                # The gradient of the whole dataset must be reset because loss.backward() will accumulate in these
                # variables. Calling zero_grad on the sample optimizers will only clear the local gradient of the
                # slice that belongs to that sample and not the global gradients of zs,nr
                zs.grad = None
                nr.grad = None

                if use_scheduler:
                    below_tresh = 0
                    for i_sched in range(index_s, index_e):
                        i_sample = i_sched - index_s
                        schedulers[i_sched].step(sample_loss_np[i_sample])
                        if shape_opts[i_sched].param_groups[0]["lr"] < 0.001:
                            # TODO manage a list of active optimizers and remove an optimizer from that list as soon
                            # as it has a too small LR
                            below_tresh += 1
                    #tqdm.write("%d / %d < 0.001 avg %0.6f" % (below_tresh, num_data, np.mean([schedulers[i].optimizer.param_groups[0]["lr"] for i in range(len(schedulers))]).item()))

                # This is only writing the gradients to zs and nr but not to the zs[i] and nr[i]
                loss.backward()

                # Copy gradients from zs and nr to the slices, then the optimizer has the gradient it needs and can make
                # a step
                for i_sched in range(index_s, index_e):
                    shape_opts[i_sched].param_groups[0]["params"][0].grad = zs.grad[i_sched]
                    shape_opts[i_sched].param_groups[0]["params"][1].grad = nr.grad[i_sched]
                    shape_opts[i_sched].step()
                    # The following is not crucial (since we override it later by coping the gradients from zs, nr),
                    # but just to be sure that no gradient accumulates,
                    # set the gradient to None. Calling shape_opt.zero_grad does not work:
                    # RuntimeError: Can't detach views in-place. Use detach() instead
                    shape_opts[i_sched].param_groups[0]["params"][0].grad = None
                    shape_opts[i_sched].param_groups[0]["params"][1].grad = None

                if self.noise_std > 0:
                    zs.data[index_s:index_e] = self.get_perturbed(zs[index_s:index_e])

                losses.append(loss.detach().cpu().numpy().tolist())

            epoch_loss = np.mean(np.array(losses))
            loss_history.append(epoch_loss)
            if self.verbose:
                if verbose and self.print_losses:
                    tqdm.write("Epoch %03d - Loss %f" % (epoch + 1, epoch_loss))
            else:
                if self.listener is not None:
                    self.listener({
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "mode": "test"
                    })

            if return_history:
                history.append({
                    "epoch": epoch + 1,
                    "l2d": epoch_results,
                    "loss": epoch_loss
                })

        self.model.train()

        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.plot(loss_history)
        #plt.savefig("/tmp/simon/plt.png")

        if return_history:
            return zs, nr, epoch_loss, history, zs_encoder, nr_encoder
        else:
            return zs, nr, epoch_loss, torch.tensor(best_zs, device=zs.device), torch.tensor(best_nr, device=nr.device), best_epoch, sample_losses, zs_encoder, nr_encoder

    def configure_for_test(self, config):
        """
        Can be used to change some of the paramters used during inference. The idea is that not for each of these params
        a new PDM has to be trained (testing is much faster)

        :param config: params to change
        """
        self.test_epochs = config["epochs"]
        self.test_optimizer = config["optimizer"]
        self.test_scheduler = config["scheduler"]
        self.test_reduction = config["test_reduction"]

        if config["conf_params"] is not None:
            self.conf_a, self.conf_b = config["conf_params"]

    # ------------------ begin PDM end2end -------------------------------------------------------

    def _run_hg_batch(self, hg, images, device):
        coords, _, _, _, _, _, var, _ = hg(images.to(device))
        confidence = self.variance2confidence(var)
        return coords, confidence

    def _run_hg_test(self, hg, data_loader, device):
        hg.eval()
        with torch.no_grad():
            hg_coords = []
            hg_confidence = []
            gt = []
            indices = []

            for batch in data_loader:
                images = batch["image"].float()
                landmarks = batch["landmarks"].float().to(device)
                idx = batch["index"].to(device)

                coords, confidence = self._run_hg_batch(hg, images, device)

                hg_coords.extend(coords.cpu().detach().numpy().tolist())
                hg_confidence.extend(confidence.cpu().detach().numpy().tolist())
                gt.extend(landmarks.cpu().detach().numpy().tolist())
                indices.extend(idx.cpu().detach().numpy().tolist())

            return torch.tensor(hg_coords, device=device).detach().requires_grad_(False), \
                   torch.tensor(hg_confidence, device=device).detach().requires_grad_(False), \
                   torch.tensor(gt, device=device).detach().requires_grad_(False), \
                   torch.tensor(indices, device=device, dtype=torch.long).detach().requires_grad_(False)

    def _run_hg_train(self, hg, data_loader, device):
        hg.train()

        for batch in data_loader:
            images = batch["image"].float()
            gt = batch["landmarks"].float().to(device)
            idx = batch["index"].to(device)

            coords, confidence = self._run_hg_batch(hg, images, device)

            yield coords, confidence, gt.detach().requires_grad_(False), idx.detach().requires_grad_(False)

    def get_optimizer(self, optimizer_params, model_params):
        optimizer_name = optimizer_params["name"]
        optimizer_params = {k:v for k,v in optimizer_params.items() if k != "name"}

        optimizers = {
            "adam" : optim.Adam,
            "sgd" : optim.SGD,
            "adadelta" : optim.Adadelta,
            "adagrad" : optim.Adagrad,
            "rmsprop" : optim.RMSprop
        }

        return optimizers[optimizer_name](params=model_params, **optimizer_params)

    def get_tasks(self, epoch, shape_init_epochs, hg_frequency, hg_pdm_together):
        if epoch < shape_init_epochs:
            # In this phase the initial zs, nr are computed. No network weights are updated, so the whole dataset
            # can be updated
            return ["SHAPE"]  # normal PDM testing

        epoch_diff = epoch - shape_init_epochs
        if epoch_diff % hg_frequency == 0:
            if hg_pdm_together:
                return ["HG", "PDM"]  # e2e training
            return ["HG"]

        return ["SHAPE", "PDM"]  # => normal PDM training

    def end2end_training(self, hg, data_loader, hg_opt_config, pdm_weight_opt_config, pdm_shape_opt_config, training_schedule=(2000, 1500, 100, False), detach_confidence=False):
        """
        3 phases:
        - 1. PDM test (zs,nr)
        - 2. HG + PDM train (zs, nr, hg, pdm)
        - 3. PDM train (zs, nr, pdm)

        Algorithm:
        1. for n_warmup epochs run PDM in test mode: learn zs, nr
        2. for n_train epochs run this every n_frequency epochs:
        2.1. learn zs, nr, HG weights, PDM weights
        2.2. update cached HG predictions after HG was updated
        3. every other epoch run this:
        3.1 learn zs, nr, PDM weights
        """

        epochs, shape_init_epochs, hg_frequency, hg_pdm_together = training_schedule

        if self.verbose:
            print("epochs", epochs)
            print("shape_init_epochs", shape_init_epochs)
            print("hg_frequency", hg_frequency)
            print("hg_pdm_together", hg_pdm_together)

        # HG cache:
        hg_coords, hg_confidence, gt, perm = self._run_hg_test(hg, data_loader, device=self.device)
        assert not hg_coords.requires_grad, "hg_coords should not require grad"
        assert not hg_confidence.requires_grad, "hg_confidence should not require grad"
        assert not gt.requires_grad, "gt should not require grad"
        assert not perm.requires_grad, "perm should not require grad"
        dirty = False

        if self.verbose:
            print("initial data loaded\n")

        lossfct = self.get_loss_function(return_sample_loss=True)
        zs, nr = self.initialize_shape_parameters(hg_coords)

        opt_hg = self.get_optimizer(hg_opt_config, hg.parameters())
        opt_pdm_weights = self.get_optimizer(pdm_weight_opt_config, self.model.parameters())
        opt_pdm_shape = self.get_optimizer(pdm_shape_opt_config, [zs, nr])

        def opt_zero_grad():
            opt_hg.zero_grad()
            opt_pdm_weights.zero_grad()
            opt_pdm_shape.zero_grad()

        if self.verbose:
            print("HG optimizer:")
            print(opt_hg, "\n")
            print("PDM weight optimizer:")
            print(opt_pdm_weights, "\n")
            print("PDM shape optimizer:")
            print(opt_pdm_shape, "\n")

        epoch_losses = []

        if self.verbose:
            epoch_iter = tqdm(range(epochs), desc="Mode: train")
        else:
            epoch_iter = range(epochs)

        epoch_loss = 0.0

        for epoch in epoch_iter:
            epoch_tasks = self.get_tasks(epoch, shape_init_epochs, hg_frequency, hg_pdm_together)
            if "HG" in epoch_tasks:
                hg.train()
            else:
                hg.eval()

            if "PDM" in epoch_tasks:
                self.model.train()
            else:
                self.model.eval()

            if "HG" not in epoch_tasks:
                if dirty: # HG was trained in last epoch ==> cached predictions are invalid
                    hg_coords, hg_confidence, gt, perm = self._run_hg_test(hg, data_loader, device=self.device)
                    dirty = False

                assert not hg_coords.requires_grad, "hg_coords should not require grad"
                assert not hg_confidence.requires_grad, "hg_confidence should not require grad"
                assert not gt.requires_grad, "gt should not require grad"
                assert not perm.requires_grad, "perm should not require grad"

                if "PDM" not in epoch_tasks: # Shape only => do all in one batch
                    iterator = [(hg_coords, hg_confidence, gt, perm)]
                else: # PDM weights have to be trained ==> respect batch size
                    iterator = []
                    batch_size = data_loader.batch_size
                    hg_coords_chunks = hg_coords.split(batch_size)
                    hg_confidence_chunks = hg_confidence.split(batch_size)
                    gt_chunks = gt.split(batch_size)
                    perm_chunks = perm.split(batch_size)
                    for chunk in zip(hg_coords_chunks, hg_confidence_chunks, gt_chunks, perm_chunks):
                        iterator.append(chunk)
            else:
                iterator = self._run_hg_train(hg, data_loader, device=self.device)
                if self.verbose:
                    iterator = tqdm(iterator)

            confidences = []
            s_losses = []
            losses = []
            for coords, confidence, groundtruth, batch_indices in iterator:
                assert coords.requires_grad == ("HG" in epoch_tasks), "coords has wrong requires_grad %s, tasks %s" % (str(coords.requires_grad), ",".join(epoch_tasks))
                assert confidence.requires_grad == ("HG" in epoch_tasks), "confidence has wrong require grad %s, tasks %s" % (str(coords.requires_grad), ",".join(epoch_tasks))
                assert not groundtruth.requires_grad, "groundtruth should not require grad"
                assert not batch_indices.requires_grad, "batch_indices should not require grad"

                if detach_confidence:
                    confidence = confidence.detach()

                confidences.extend(confidence.cpu().detach().numpy().tolist())

                opt_zero_grad()

                l2d, l3d = self.forward(zs[batch_indices], nr[batch_indices])

                # TODO make configurable which loss when
                loss_sum = 0
                sample_losses = torch.zeros(batch_indices.shape, requires_grad=False)
                if "HG" in epoch_tasks:
                    hg_loss, hg_sample_losses = lossfct(coords, groundtruth)
                    hg_loss.backward(retain_graph=("SHAPE" in epoch_tasks or "PDM" in epoch_tasks))
                    opt_hg.step()
                    dirty = True
                    loss_sum += hg_loss.detach().data.item()
                    sample_losses += hg_sample_losses.detach().cpu()
                    opt_zero_grad()

                if "SHAPE" in epoch_tasks:
                    pdm_reconstruction_loss, sample_reconstr_losses = lossfct(l2d * confidence, coords * confidence)
                    pdm_reconstruction_loss.backward(retain_graph=("PDM" in epoch_tasks or "HG" in epoch_tasks))
                    opt_pdm_shape.step()
                    loss_sum += pdm_reconstruction_loss.detach().data.item()
                    sample_losses += sample_reconstr_losses.detach().cpu()
                    opt_zero_grad()

                if "PDM" in epoch_tasks or "HG" in epoch_tasks:
                    end_loss, end_sample_losses = lossfct(l2d, groundtruth)
                    end_loss.backward()
                    opt_pdm_weights.step()
                    loss_sum += end_loss.detach().data.item()
                    sample_losses += end_sample_losses.detach().cpu()
                    opt_zero_grad()

                s_losses.extend(sample_losses.numpy())

                zs.data[batch_indices] = self.get_perturbed(zs[batch_indices])

                losses.append(loss_sum)

            s_losses = np.array(s_losses)

            epoch_loss = np.mean(s_losses).item()
            epoch_losses.append(epoch_loss)
            if self.verbose:
                tqdm.write("Epoch %03d - Train loss %f - Mean Confidence %f -  Tasks %s " % (epoch + 1, epoch_loss, np.mean(confidences).item(), ",".join(epoch_tasks)))
            else:
                if self.listener is not None:
                    self.listener({ # TODO log sample_reconstr losses
                        "mode": "e2e",
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "l2d_coords": l2d.detach().cpu().numpy().tolist(),
                        "gt_coords": groundtruth.detach().cpu().numpy().tolist(),
                        "hg_coords": coords.detach().cpu().numpy().tolist(),
                        "pdm": self,
                        "hg": hg,
                        "location": self.device
                    })

        return zs, nr, epoch_losses
