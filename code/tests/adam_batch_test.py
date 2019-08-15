import sys
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

random_seed = 0
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

dev = torch.device(sys.argv[1])

model = torch.nn.Linear(32,16)
model = model.to(dev)
model = model.double()
model.eval()

lossfct = torch.nn.L1Loss(reduction='sum')

n_samples = 128
opts = []
samples = []
for i in range(n_samples):
    # create independent samples and ground truth
    x = torch.randn(32,device=dev, dtype=torch.float64).requires_grad_(True)
    y = torch.randn(16,device=dev, dtype=torch.float64).requires_grad_(False)
    samples.append((x,y))

    # create one optimizer for each sample and only add x as parameter to optimize
    opts.append(torch.optim.Adam(params=[x], lr=0.01))

epochs = int(sys.argv[2])
bs = int(sys.argv[3])

for epoch in range(epochs):
    for i in range(n_samples // bs + 1):
        start, end = i*bs, min((i+1)*bs, n_samples)
        if end <= start:
            continue

        for opt in opts:
            opt.zero_grad()

        # in order to be able to process multiple samples at one time (faster training), I stack the samples of a
        # batch, but they should still be independent from each other, i.e. their optimizers should not interact,
        # neither should their gradients affect each other
        batch_in = torch.stack([samples[k][0] for k in range(start, end)])
        batch_gt = torch.stack([samples[k][1] for k in range(start, end)])

        pred = model(batch_in)
        loss = lossfct(pred, batch_gt)
        loss.backward()

        for i in range(start, end):
            #opts[i].param_groups[0]["params"][0].grad =opts[i].param_groups[0]["params"][0].grad*0 + 1/(i+1)
            opts[i].step()

all_x = torch.stack([samples[k][0] for k in range(n_samples)])
all_y = torch.stack([samples[k][1] for k in range(n_samples)])
pred = model(all_x)
loss = lossfct(pred,all_y)
print(loss.data.item() / len(all_x))