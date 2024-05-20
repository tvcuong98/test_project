from qm7 import QM7,QM7b
import torch.nn.functional as F
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import random
import torch.nn as nn
from models.GCN import GCNConv
from models.GraphSage import GraphSage,GraphSageRegression
import os
import glob
from display_utils import display_dataset_discription
from disk_utils import delete_all_files_in_folder
from data_utils import pre_normalize_feature_label
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
    model = GraphSageRegression(4,4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()
    fold_train = QM7(root=root,train=True,split=fold)
    fold_test = QM7(root=root,train=False,split=fold)
    print(f'Number of graphs: {len(fold_train)}')
    print(f'Number of graphs: {len(fold_test)}')


    train_loader = torch_geometric.loader.DataLoader(fold_train, batch_size=1024, shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(fold_test, batch_size=1024, shuffle=False)
    for epoch in range(1000):
        model.train()
        running_train_loss = 0.0
        step =0
        for step, data in enumerate(train_loader):
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss =nn.MSELoss()(out, data.y)
            running_train_loss+=loss
            loss.backward()
            optimizer.step()
            step+=1
        epoch_train_loss = running_train_loss/step
        if epoch%1 == 0:
            step=0
            model.eval()
            running_test_loss=0.0
            for step, data in enumerate(test_loader):
                data.to(device)
                out = model(data)
                loss = nn.MSELoss()(out,data.y)
                running_test_loss+=loss
                step+=1
            epoch_test_loss = running_test_loss/step
            print(f"Epoch_{epoch}: train_loss_{epoch_train_loss} test_loss_{epoch_test_loss}")
    model.eval()
    running_val_loss = []
    running_val_mse =[]
    running_val_rmse =[]
    running_val_r2 =[]
    running_val_mae =[]
    for step, data in enumerate(test_loader):
        data.to(device)
        optimizer.zero_grad()
        out = model(data).cpu().detach()
        label = data.y.cpu().detach()
        val_loss = nn.MSELoss()(out, label)
        running_val_loss.append(val_loss)
        running_val_mse.append(mean_squared_error(label, out))
        running_val_rmse.append(np.sqrt(mean_squared_error(label, out)))
        running_val_r2.append(r2_score(label, out))
        running_val_mae.append(mean_absolute_error(label,out))

    print(f"  MSE LOSS VAL: {np.mean(running_val_loss):.4f}")
    print(f"  Mean Squared Error: {np.mean(running_val_mse):.4f}")
    print(f"  Root Mean Squared Error: {np.mean(running_val_rmse):.4f}")
    print(f"  R-squared: {np.mean(running_val_r2):.4f}")
    print(f"  Mean Absolute Error: {np.mean(running_val_mae):.4f}")

