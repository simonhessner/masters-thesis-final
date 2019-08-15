import torch
import torch.nn as nn
import json
import os
from common.early_stopping import EarlyStopping
from collections import OrderedDict
import argparse
from common.util import mkdir_if_not_exists
from common.metrics import NO_OUTLINE_MASK


class Encoder(nn.Module):
    def __init__(self, zs_size, nr_size, n_coords=68):
        super(Encoder, self).__init__()

        self.zs_size = zs_size
        self.nr_size = nr_size

        hidden_size = 1024

        self.layers = nn.Sequential(OrderedDict([
            ('lin1',  nn.Linear(in_features=2*n_coords, out_features=hidden_size)),
            ('relu1', nn.ReLU()),
            ('lin3',  nn.Linear(in_features=hidden_size, out_features=self.zs_size + self.nr_size)),
        ]))

        self.n_coords = n_coords

    def forward(self, x):
        # Note: The result of the encoder differs on TITAN X and K40 if no double precision is used. It is just commented
        # out because we want to reproduce gridsearch results. Might be a good idea to change this some day
        #x = x.double()
        #self.layers = self.layers.double()
        x = self.layers(x)
        return x.float()

    def get_separated(self, x):
        zs = x[:, :self.zs_size]
        nr = x[:, self.zs_size:]
        return zs, nr


def train(train_data, test_data, is_49lm=False):
    if is_49lm:
        n_coords = 49
        x = torch.tensor(train_data["coords"], device=device)[:,NO_OUTLINE_MASK,:].view(-1, n_coords * 2)
    else:
        n_coords = 68
        x = torch.tensor(train_data["coords"], device=device).view(-1, n_coords * 2)

    y_zs = torch.tensor(train_data["zs"], device=device)
    y_nr = torch.tensor(train_data["nr"], device=device)

    bs = 32
    epochs = 250
    n = x.shape[0]

    enc = Encoder(zs_size=y_zs.shape[1], nr_size=y_nr.shape[1],n_coords=n_coords).to(device)
    enc.train()

    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(enc.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=15, verbose=True, threshold=10e-3)
    should_stop = EarlyStopping(patience=int(scheduler.patience * 2.5), max_ratio=0.995)

    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n)

        for i in range(n//bs + 1):
            optimizer.zero_grad()
            start, end = i*bs, (i+1)*bs
            batch_idxes = perm[start:end]

            batch = x[batch_idxes]
            if batch.shape[0] < 1:
                continue

            pred = enc(batch)
            pred_zs, pred_nr = enc.get_separated(pred)
            gt_zs = y_zs[batch_idxes]
            gt_nr = y_nr[batch_idxes]

            loss = mse(pred_zs, gt_zs) + mse(pred_nr, gt_nr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item() * pred_nr.shape[0]

        epoch_loss /= n

        testloss = test(enc, test_data, is_49lm=is_49lm)
        if should_stop(testloss):
            print("Early stopping")
            break
        scheduler.step(testloss)

        if epoch % 50 == 0:
            print("Epoch %d | %d LMs | Train loss %0.5f | Test loss %0.5f" % (epoch, n_coords, epoch_loss, testloss))

    return enc


def test(encoder, test_data, is_49lm=False):
    encoder.eval()

    if is_49lm:
        n_coords = 49
        x = torch.tensor(test_data["coords"], device=device)[:,NO_OUTLINE_MASK,:].view(-1, n_coords * 2)
    else:
        n_coords = 68
        x = torch.tensor(test_data["coords"], device=device).view(-1, n_coords * 2)

    y_zs = torch.tensor(test_data["zs"], device=device)
    y_nr = torch.tensor(test_data["nr"], device=device)

    n = x.shape[0]
    bs = 512

    mse = nn.MSELoss()

    loss = 0.0

    with torch.no_grad():
        for i in range(n // bs + 1):
            start, end = i * bs, (i + 1) * bs

            batch = x[start:end]
            if batch.shape[0] < 1:
                continue

            pred = encoder(batch)
            pred_zs, pred_nr = encoder.get_separated(pred)
            gt_zs = y_zs[start:end]
            gt_nr = y_nr[start:end]

            loss += (mse(pred_zs, gt_zs) + mse(pred_nr, gt_nr)).data.item()  * pred_nr.shape[0]

        encoder.train()
    loss /= n
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("zs_and_nr_from_pdm", type=str, help="json file with zs, nr and coords after PDM training")
    parser.add_argument("target", type=str, help="Where to store encoder (.torch)")
    parser.add_argument("--is_49lm", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 = CPU")
    args = parser.parse_args()

    device = torch.device("cuda:%d" % args.gpu if args.gpu >= 0 else "cpu")
    #print(device)
    data = json.load(open(args.zs_and_nr_from_pdm, "r"))

    enc = train(data["train"], data["test"], is_49lm=args.is_49lm)
    test_loss = test(enc, data["test"], is_49lm=args.is_49lm)
    print("Final test loss", test_loss)
    mkdir_if_not_exists(os.path.dirname(args.target))


    torch.save({
        "state_dict" : enc.state_dict(),
        "zs_size" : enc.zs_size,
        "nr_size" : enc.nr_size
    }, open(args.target, "wb"))