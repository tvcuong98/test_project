import torch
from torch_geometric.data import Data

def pre_normalize_feature_label(data):
    x_mean, x_std = 1.6812959, 6.700323
    y_mean, y_std = -1538.0377, 223.91891
    data.x = (torch.Tensor(data.x) - x_mean) / x_std
    data.y = (torch.Tensor(data.y) - y_mean) / y_std
    return data