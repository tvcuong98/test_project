from qm7 import QM7,QM7b
import torch.nn.functional as F
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import random
import torch.nn as nn
from models.GAT import GAT
import os
import glob
from display_utils import display_dataset_discription
from disk_utils import delete_all_files_in_folder

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


for fold in range(5):
    model = GAT(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()
    fold_train = QM7(root=root,train=True,split=fold)
    fold_test = QM7(root=root,train=False,split=fold)


    train_loader = DataLoader(fold_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(fold_test, batch_size=64, shuffle=False)
    model.train()
    for epoch in range(100):
        running_train_loss = 0.0
        for step, data in enumerate(train_loader):
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            label = data.y
            train_loss = nn.MSELoss()(out, data.y)
            running_train_loss+=train_loss
            train_loss.backward()
            optimizer.step()
        print(f"Epoch_{epoch}: loss_{running_train_loss/step}")
    model.eval()
    running_val_loss = 0.0
    for step, data in enumerate(test_loader):
        data.to(device)
        optimizer.zero_grad()
        out = model(data)
        label = data.y
        val_loss = nn.MSELoss()(out, label)
        running_val_loss+=val_loss
    print(running_val_loss/step)