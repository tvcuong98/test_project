import torch
from models.egnn_pytorch import EGNN
import torch.nn as nn
from qm7 import QM7_pytorch_dataset,QM7
from torch_geometric.utils import to_dense_batch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import os
output_dir = "my-graph/output_EGNN"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def plotting_parity(T_pred,T_test,fold,name="EGNN"):
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
class EGNNRegression(nn.Module):
    def __init__(self, in_channels):
        
        super().__init__()
        self.egnn1 =EGNN(dim=in_channels)
        self.egnn2 =EGNN(dim=in_channels)
        self.egnn3 =EGNN(dim=in_channels)
        self.egnn4 =EGNN(dim=in_channels)
        self.linear=nn.Linear(in_channels,1)
 
         
    def forward(self, feats, coors):      
        '''
        ''' 
        # feats = feats.unsqueeze(-1)
        feats,coors= self.egnn1(feats,coors)
        feats,coors= self.egnn2(feats,coors)
        feats,coors= self.egnn2(feats,coors)
        feats,coors= self.egnn2(feats,coors)
        feats = feats.mean(dim=1) # it is now batch, out_channels
        out = self.linear(feats)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"


for fold in range(5):
    mse_list =[]
    rmse_list = []
    r2_list=[]
    mae_list=[]
    model = EGNNRegression(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()
    ds_train = QM7_pytorch_dataset('qm7.mat', train=True, split=fold)
    ds_test = QM7_pytorch_dataset('qm7.mat', train=False, split=fold)


    train_loader = DataLoader(ds_train, batch_size=1024, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=1024, shuffle=False)
    model.train()
    for epoch in range(1000):
        running_train_loss = 0.0
        for step, data in enumerate(train_loader):
            feature, label = data
            feature, label = feature.to(device), label.to(device)
            feats = feature[:,:,:]
            coors = feature[:,:,:-1]
            optimizer.zero_grad()
            out = model(feats,coors)
            train_loss = nn.MSELoss()(out, label)
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
        feature, label = data
        feature, label = feature.to(device), label.to(device)
        feats = feature[:,:,:]
        coors = feature[:,:,:-1]
        optimizer.zero_grad()
        out = model(feats,coors).cpu().detach()
        label = label.cpu().detach()
        # Append outputs and labels to the lists
        all_outputs.append(out.numpy())
        all_labels.append(label.numpy())


        val_loss = nn.MSELoss()(out, label)
        running_val_loss.append(val_loss)
        running_val_mse.append(mean_squared_error(label, out))
        running_val_rmse.append(np.sqrt(mean_squared_error(label, out)))
        running_val_r2.append(r2_score(label, out))
        running_val_mae.append(mean_absolute_error(label,out))
    print("EGNN Fold :",fold)
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
    plotting_parity(all_outputs,all_labels,fold,name="EGNN")

print(f"\n Final:")
print(f"  Mean Squared Error: {np.mean(mse_list):.4f}")
print(f"  Root Mean Squared Error: {np.mean(rmse_list):.4f}")
print(f"  R-squared: {np.mean(r2_list):.4f}")
print(f"  Mean Absolute Error: {np.mean(mae_list):.4f}")
        