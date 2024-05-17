from ds import QM7
import torch
from torch import nn
import numpy as np

from ml_models import LinearRegression,LassoRegression

def mae(y_pred, y_true):
    return torch.abs(y_pred - y_true).mean()

def train(model, split):
    ds_train = QM7('qm7.mat', train=True, split=split)
    ds_test = QM7('qm7.mat', train=False, split=split)

    model = model
    # model.train()

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)


    x = torch.Tensor(ds_train.X)
    y = torch.Tensor(ds_train.y)
    x_mean, x_std = 1.6812959, 6.700323
    y_mean, y_std = -1538.0377, 223.91891


    x = torch.Tensor(ds_train.X)
    y = torch.Tensor(ds_train.y)
    
    x_test = torch.Tensor(ds_test.X)
    y_test = torch.Tensor(ds_test.y)

    scaler = torch.cuda.amp.GradScaler()

    ep = 1000
    for e in range(ep):
        # for x, y in loader:
        with torch.cuda.amp.autocast():
            x = x
            y = y
            y_pred = model(x)
            loss = nn.MSELoss()(y_pred, y)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        result = mae(model(x_test), y_test)*y_std
    return result

if __name__ == "__main__":
    maes = [train(LinearRegression,i).detach().cpu().numpy() for i in range(5)]
    print(maes)
    print(np.mean(maes))