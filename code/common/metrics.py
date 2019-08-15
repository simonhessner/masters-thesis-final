import torch
import numpy as np
from torch import nn
from common.util import move2device


class WingLoss(nn.Module):
    """
    Idea and formula: https://arxiv.org/pdf/1711.06753.pdf
    """

    def __init__(self, width=5, curvature=0.5, return_sample_loss=False, reduction='mean'):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)
        self.return_sample_loss = return_sample_loss

        if reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be 'mean' or 'sum', you gave '%s'" % reduction)
        self.reduction = reduction

    def forward(self, prediction, target):
        diff = target - prediction
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        #print("%f - %f" % (torch.min(prediction).data.item(), torch.max(prediction).data.item()), "%f - %f" % (torch.min(target).data.item(), torch.max(target).data.item()))
        #print("%f - %f" % (torch.min(diff_abs).data.item(), torch.max(diff_abs).data.item()))

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        #print(torch.min(idx_smaller), torch.max(idx_smaller))
        #print(torch.min(idx_bigger), torch.max(idx_bigger))
        #print("smaller", torch.sum(idx_smaller).data.item(), " / bigger", torch.sum(idx_bigger).data.item())

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger]  = loss[idx_bigger] - self.C

        sample_grouped = loss.view(loss.shape[0], -1)
        sample_loss = sample_grouped.mean(dim=1)

        if self.reduction == "mean":
            scalar_loss = sample_loss.mean()
        else:
            scalar_loss = sample_loss.sum()

        if self.return_sample_loss:
            return scalar_loss, sample_loss
        return scalar_loss


def norm(v, dim=None):
    """
    Calculates the root-mean-square error (RMSE) for each landmark.

    :param v: vector where a certain dimension should be converted to its RMSE

    :param dim: dimension
    :return: Tensor with the RMSE for each landmark
    """
    # print(v.shape) # BS x N_LM x 2  OR  BS x 2 when called from get_iod
    r = torch.sqrt(torch.sum(torch.pow(v,2), dim=dim))
    # print(r.shape) # BS x N_LM  OR  BS when called from get_iod
    return r


def mask(x,y,return_count=False):
    # Hidden landmarks will have -10. as x or y coordinates
    # If at least one coordinate (x or y) is hidden, the minimum will be 0 (since torch.ne = 0 when coord = -10)
    # If both are hidden, the minimum is 0.
    # If none is hidden, the minumum is 1
    M,_ = torch.min(torch.ne(y, -10.).float(), dim=2) # the second return value is the index of the minimum
    #print(M.shape) # BS x (49 or 68, depending on with or without outline)

    if return_count:
        # Multiplying x with the mask sets the value of x to 0 if the mask is 0, otherwise it does not change it
        return torch.mul(x,M), torch.sum(M, dim=1) # the second parameter is the number of non-hidden LMs per sample in the batch
    return torch.mul(x,M)


"""
Explanation to the formulas

The metric comes from equation 1 in this paper:
https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w25/papers/Shen_The_First_Facial_ICCV_2015_paper.pdf

It differs from the new formula eq 17 on p 12 from https://link.springer.com/content/pdf/10.1007%2Fs11263-018-1134-y.pdf
by the factor 1/N_L vs 1/sqrt(N_L)

In our paper, we use the older version without the sqrt. The sqrt version is used in the newer Menpo challenge.
"""

# https://link.springer.com/content/pdf/10.1007%2Fs11263-018-1134-y.pdf page 6 figure 3 a
LEFT_EYE_CORNER_IDX = 36  # landmark number 37 (because 1-indexed)
RIGHT_EYE_CORNER_ID = 45  # landmark_number 46 (because 1-indexed)

NO_OUTLINE_MASK = list(set(range(17,68))-{60,64})


def get_iod(test_y):
    return norm(test_y[:, LEFT_EYE_CORNER_IDX, :] - test_y[:, RIGHT_EYE_CORNER_ID, :], dim = 1)


def get_wrapped(t):
    """
    If t is a tensor with only 49 landmarks per sample, it will be wrapped in a new tensor with 68 landmarks where
    the missing (= outline) landmarks are all set to -10.0

    This is needed to calculate the "without outline" error because it relies on fixed indices for the eyes.
    Another solution would be to hardcode the indices for both cases (49 or 68 landmarks)
    """
    if t.shape[1] == 68:
        return t
    else:
        tmp = torch.ones((t.shape[0], 68, t.shape[2]), device=t.device) * -10.0
        tmp[:, NO_OUTLINE_MASK, :] = t
        return tmp


def with_outline_error(pred_y, test_y):
    #print(test_y.shape) # BS x N_LM x 2
    #print(test_y[:, LEFT_EYE_CORNER_IDX, :].shape) # BS x 2
    #print(interocular_distance.shape) # BS (one float per sample)

    # handle 49 and 69 LMs
    pred_y = get_wrapped(pred_y)
    test_y = get_wrapped(test_y)

    m, n = mask(norm(pred_y - test_y, dim=2), test_y, True)
    # print(m.shape, n.shape)  # torch.Size([32, 49]) torch.Size([32])
    # Calculate the average of the RMSE of each landmark
    landmark_dist_sum = torch.sum(m, dim=1)/n

    # normalize by IOD
    return landmark_dist_sum / get_iod(test_y)


def without_outline_error(pred_y, test_y):
    # According to Yao-Chong most papers remove also LM 61 and 65 (inner mouth corner) for the "without outline" case
    # See Mail from April 18th:
    # It seems the 49 came from here (1) https://ri.cmu.edu/pub_files/2013/5/main.pdf.
    # They based the 49 landmarks on a 66-landmark annotation, which seems to have come from
    # (2) https://ieeexplore.ieee.org/document/5995618 or
    # (3) https://www.ri.cmu.edu/pub_files/2009/9/CameraReady-6.pdf.
    # I can't find an image that explicitly shows the landmark points with numbers,
    # but the images in (2) show that the landmarks in the inner corners of the mouth are missing.

    # handle 49 and 69 LMs
    pred_y = get_wrapped(pred_y)
    test_y = get_wrapped(test_y)

    # Contains the RMSE of each landmark in the back
    m, n = mask(norm(pred_y[:,NO_OUTLINE_MASK,:] - test_y[:,NO_OUTLINE_MASK,:], dim=2), test_y[:,NO_OUTLINE_MASK,:], True)
    # print(m.shape, n.shape)  # torch.Size([32, 49]) torch.Size([32])
    landmark_dist_sum = torch.sum(m, dim=1)/n
    return landmark_dist_sum / get_iod(test_y)


def evaluate(pred, gt):
    n_lm = pred.shape[1]
    if n_lm == 68:
        return {
            "with_outline" : np.median(100 * with_outline_error(pred, gt).cpu().data.numpy()).item(),
            "without_outline" : np.median(100 * without_outline_error(pred, gt).cpu().data.numpy()).item()
        }
    elif n_lm == 49:
        return {
            "with_outline": 10000000.0,
            "without_outline": np.median(100 * without_outline_error(pred, gt).cpu().data.numpy()).item()
        }
    else:
        raise ValueError("Only 49 or 68 LM supported")


def evaluate_menpo(pred, gt):
    assert gt.shape[1] == 68

    gt_min_x = gt[:,:,0].min(dim=1)[0]
    gt_max_x = gt[:,:,0].max(dim=1)[0]
    gt_min_y = gt[:,:,1].min(dim=1)[0]
    gt_max_y = gt[:,:,1].max(dim=1)[0]
    width = gt_max_x - gt_min_x
    height = gt_max_y - gt_min_y
    normalization = (width + height) / 2
    #normalization = (width.pow(2) + height.pow(2)).sqrt()

    n_lm = pred.shape[1]

    pred = get_wrapped(pred)
    gt = get_wrapped(gt)

    def norm_err(lm_mask):
        # norm: euclidean distance for each LM, error is the average distance for all landmarks
        error = torch.sum(norm(pred[:,lm_mask] - gt[:,lm_mask], dim=2), dim=1) / len(lm_mask)
        norm_err = error / normalization
        return np.median(100 * norm_err.cpu().data.numpy()).item()

    if n_lm == 68:
        outline_error = norm_err(list(range(68)))
    else:
        outline_error = 1000000.0
    no_outline_error = norm_err(NO_OUTLINE_MASK)

    return outline_error, no_outline_error


def benchmark(model, easy, hard, location):
    model.eval()  # Eval mode

    easy_woutline = []
    easy_noutline = []
    hard_woutline = []
    hard_noutline = []

    with torch.no_grad():
        for sample in easy:
            x = move2device(sample['image'].float(), location)
            y = move2device(sample['landmarks'].float(), location)
            coords, *rest_easy = model(x)

            if isinstance(coords, list):
                coords = coords[-1]  # get last hourglass output in case of intermediate supervision

            if coords.shape[1] == 68:
                easy_woutline.append(with_outline_error(coords, y).data.cpu().numpy())
                easy_noutline.append(without_outline_error(coords, y).data.cpu().numpy())
            else:
                easy_noutline.append(without_outline_error(coords, y).data.cpu().numpy())
                easy_woutline.append(np.full_like(easy_noutline[-1], 100000.0))  # 100000.0 is a dummy value

        for sample in hard:
            x = move2device(sample['image'].float(), location)
            y = move2device(sample['landmarks'].float(), location)
            coords, *rest_hard = model(x)

            if isinstance(coords, list):
                coords = coords[-1]  # get last hourglass output in case of intermediate supervision

            if coords.shape[1] == 68:
                hard_woutline.append(with_outline_error(coords, y).data.cpu().numpy())
                hard_noutline.append(without_outline_error(coords, y).data.cpu().numpy())
            else:
                hard_noutline.append(without_outline_error(coords, y).data.cpu().numpy())
                hard_woutline.append(np.full_like(hard_noutline[-1], 100000.0))  # 100000.0 is a dummy value

    model.train()  # Back to train model

    res = {
        "e68": np.median(100 * np.concatenate(easy_woutline)).item(),  #item() converts numpy dtype to python dtype (important e.g. for json.dump())
        "h68": np.median(100 * np.concatenate(hard_woutline)).item(),
        "e49": np.median(100 * np.concatenate(easy_noutline)).item(),
        "h49": np.median(100 * np.concatenate(hard_noutline)).item()
    }

    return res
