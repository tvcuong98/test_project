from qm7 import QM7,QM7b
import torch.nn.functional as F
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import random
import torch.nn as nn
from models.GCN import GCNConv
import os
import glob
from display_utils import display_dataset_discription
from disk_utils import delete_all_files_in_folder
from data_utils import pre_normalize_feature_label
### Dealing with dataset:
root = "/home/edabk/cuong/test_project/my-graph/datasets/qm7"
processed_path="/home/edabk/cuong/test_project/my-graph/datasets/qm7/processed"
delete_all_files_in_folder(processed_path)
# dataset = QM7(root=root)
# display_dataset_discription(dataset)
# ### Spliting the dataset:
# fold_split = dataset.fold_split # (5,1433)
# data_fold_train = []
# data_fold_test = []
# for fold in range(5):
#     split = fold_split[fold] # (1433,)
#     test_mask = torch.zeros(len(dataset), dtype=bool) # (7165,), all False
#     test_mask[split]=True # turn on True for the sample that have its index listed in split
#     # the mask is now still (7165,), the True value are for test dataset
#     testset = dataset[test_mask]
#     train_mask = ~test_mask
#     trainset = dataset[train_mask]
#     data_fold_train.append(trainset)
#     data_fold_test.append(testset)
#     print(f'fold_{fold}')
#     print(f'Number of training graphs: {len(trainset)}')
#     print(f'Number of test graphs: {len(testset)}')
# display_dataset_discription(data_fold_train[2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

model = GCNConv(4,1,32,final=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

for fold in range(5):
    model.train()
    fold_train = QM7(root=root,train=True,split=fold)
    fold_test = QM7(root=root,train=False,split=fold)
    print(f'Number of graphs: {len(fold_train)}')
    print(f'Number of graphs: {len(fold_test)}')


    train_loader = torch_geometric.loader.DataLoader(fold_train, batch_size=64, shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(fold_test, batch_size=64, shuffle=False)
    for epoch in range(100000):
        model.train()
        for step, data in enumerate(train_loader):
            data.to(device)
            # print(data.ptr)
            # print(data.batch)
            optimizer.zero_grad()
            out = model(data)
            # print(out.shape)
            loss =nn.L1Loss()(out, data.y)
            loss.backward()
            optimizer.step()
        if epoch%1 == 0:
            print(f"train:{loss}")
            model.eval()
            running_test_loss=0.0
            for step, data in enumerate(test_loader):
                data.to(device)
                out = model(data)
                loss = nn.L1Loss()(out,data.y)
                running_test_loss+=loss
            print(f"test:{running_test_loss}")

