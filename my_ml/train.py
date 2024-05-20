from ds import QM7
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt

device="cuda"
# data preprocessing
from sklearn.model_selection import train_test_split

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# models 
from ml_function import LinearRegression,RidgeRegression

output_dir = 'my_ml/output_ml'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def plotting_parity(T_pred,T_test,model_name,fold):
    plt.figure(figsize=(10, 6))
    plt.scatter(T_test, T_pred, color='green', label='Predicted vs Actual')
    plt.plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], 'r--', label='Perfect Prediction')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_fold_{fold}_parity.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()
def train_ml(name,model, split):
    ds_train = QM7('qm7.mat', train=True, split=split)
    ds_test = QM7('qm7.mat', train=False, split=split)


    x = torch.Tensor(ds_train.X)
    y = torch.Tensor(ds_train.y)
    
    x_test = torch.Tensor(ds_test.X)
    y_test = torch.Tensor(ds_test.y)

    if name=="Gaussian Kernel Regression":
        y_test_pred = model(x,y,x_test,kernel_func='linear').cpu()
    else:
        model.fit(x,y)
        y_test_pred = model(x_test).cpu()
    y_test = y_test.cpu()
    mse = mean_squared_error(y_test, y_test_pred) 
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred) 
    plotting_parity(T_pred=y_test_pred,T_test=y_test,model_name=name,fold=split)
    return mse,rmse,r2,mae

models = {# "Linear Regression":LinearRegression(input_dim=276,device=device),
          "Ridge Regression":RidgeRegression(input_dim=276,device=device)}

for name, model in models.items():
    mse_list =[]
    rmse_list = []
    r2_list=[]
    mae_list=[]
    for fold in range(5):
        mse,rmse,r2,mae = train_ml(name,model,fold)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        mae_list.append(mae)
    print(f"\n{name}:")
    print(f"  Mean Squared Error: {np.mean(mse_list):.4f}")
    print(f"  Root Mean Squared Error: {np.mean(rmse_list):.4f}")
    print(f"  R-squared: {np.mean(r2_list):.4f}")
    print(f"  Mean Absolute Error: {np.mean(mae_list):.4f}")