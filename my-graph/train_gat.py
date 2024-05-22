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
import numpy as np
import matplotlib.pyplot as plt
from display_utils import display_dataset_discription
from disk_utils import delete_all_files_in_folder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
### Dealing with dataset:
root = "my-graph/datasets/qm7"
processed_path="my-graph/datasets/qm7/processed"
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
output_dir = "my-graph/output_GAT"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def plotting_parity(T_pred,T_test,fold,name="GAT"):
    plt.figure(figsize=(6, 6))
    plt.scatter(T_test, T_pred, color='green', label='Predicted vs Actual')
    plt.plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], 'r--', label='Perfect Prediction')
    plt.title(f'{name} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f'{name.replace(" ", "_")}_fold_{fold}_parity.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"


for fold in range(5):
    mse_list =[]
    rmse_list = []
    r2_list=[]
    mae_list=[]
    model = GAT(4,4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()
    fold_train = QM7(root=root,train=True,split=fold)
    fold_test = QM7(root=root,train=False,split=fold)


    train_loader = DataLoader(fold_train, batch_size=1024, shuffle=True)
    test_loader = DataLoader(fold_test, batch_size=1024, shuffle=False)
    model.train()
    for epoch in range(1000):
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
        # print(f"Epoch_{epoch}: loss_{running_train_loss/step}")
    model.eval()
    running_val_loss = []
    running_val_mse =[]
    running_val_rmse =[]
    running_val_r2 =[]
    running_val_mae =[]
    all_outputs = []
    all_labels = []
    for step, data in enumerate(test_loader):
        data.to(device)
        optimizer.zero_grad()
        out = model(data).cpu().detach()
        label = data.y.cpu().detach()
        # Append outputs and labels to the lists
        all_outputs.append(out.numpy())
        all_labels.append(label.numpy())

        val_loss = nn.MSELoss()(out, label)
        running_val_loss.append(val_loss)
        running_val_mse.append(mean_squared_error(label, out))
        running_val_rmse.append(np.sqrt(mean_squared_error(label, out)))
        running_val_r2.append(r2_score(label, out))
        running_val_mae.append(mean_absolute_error(label,out))
    print("GAT Fold :",fold)
    print(f"  MSE LOSS VAL: {np.mean(running_val_loss):.4f}")
    print(f"  Mean Squared Error: {np.mean(running_val_mse):.4f}")
    print(f"  Root Mean Squared Error: {np.mean(running_val_rmse):.4f}")
    print(f"  R-squared: {np.mean(running_val_r2):.4f}")
    print(f"  Mean Absolute Error: {np.mean(running_val_mae):.4f}")

    mse_list.append(np.mean(running_val_mse))
    rmse_list.append(np.mean(running_val_rmse))
    r2_list.append(np.mean(running_val_r2))
    mae_list.append(np.mean(running_val_mae))

    # Concatenate all the elements in each list into a single NumPy array, to plot
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    plotting_parity(all_outputs,all_labels,fold,name="GAT")

print(f"\n Final:")
print(f"  Mean Squared Error: {np.mean(mse_list):.4f}")
print(f"  Root Mean Squared Error: {np.mean(rmse_list):.4f}")
print(f"  R-squared: {np.mean(r2_list):.4f}")
print(f"  Mean Absolute Error: {np.mean(mae_list):.4f}")