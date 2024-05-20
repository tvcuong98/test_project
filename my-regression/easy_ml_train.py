from ds import QM7
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
# data preprocessing
from sklearn.model_selection import train_test_split

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
# models 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# def mae(y_pred, y_true):
#     return torch.abs(y_pred - y_true).mean()

def plotting_parity(T_pred,T_test,model_name,fold):
    plt.figure(figsize=(10, 6))
    plt.scatter(T_test, T_pred, color='green', label='Predicted vs Actual')
    plt.plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], 'r--', label='Perfect Prediction')
    plt.title(f'{name} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_fold_{fold}_parity.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()
def train(name,model, split):
    ds_train = QM7('qm7.mat', train=True, split=split)
    ds_test = QM7('qm7.mat', train=False, split=split)

    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # x_mean, x_std = 1.6812959, 6.700323
    # y_mean, y_std = -1538.0377, 223.91891

    x_mean, x_std = 0, 1
    y_mean, y_std = 0,1


    x = (torch.Tensor(ds_train.X).cuda() - x_mean) / x_std
    y = (torch.Tensor(ds_train.y).cuda() - y_mean) / y_std
    
    x_test = (torch.Tensor(ds_test.X).cuda() - x_mean) / x_std
    y_test = (torch.Tensor(ds_test.y).cuda() - y_mean) / y_std

    scaler = torch.cuda.amp.GradScaler()

    ep = 1000
    for e in range(ep):
        # for x, y in loader:
        with torch.cuda.amp.autocast():
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            loss = nn.MSELoss()(y_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        y_test_pred= model(x_test)
    y_test = y_test.cpu()
    y_test_pred = y_test_pred.cpu()
    mse = mean_squared_error(y_test, y_test_pred) * y_std
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred) * y_std

    plotting_parity(T_pred=y_test_pred,T_test=y_test,model_name=name,fold=split)
    return mse,rmse,r2,mae

def train_ml(name,model, split):
    ds_train = QM7('qm7.mat', train=True, split=split)
    ds_test = QM7('qm7.mat', train=False, split=split)


    # x_mean, x_std = 1.6812959, 6.700323
    # y_mean, y_std = -1538.0377, 223.91891

    x_mean, x_std = 0, 1
    y_mean, y_std = 0,1

    x = (torch.Tensor(ds_train.X) - x_mean) / x_std
    y = (torch.Tensor(ds_train.y) - y_mean) / y_std
    
    x_test = (torch.Tensor(ds_test.X) - x_mean) / x_std
    y_test = (torch.Tensor(ds_test.y) - y_mean) / y_std


    model.fit(x,y)
    y_test_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred) * y_std
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred) * y_std
    plotting_parity(T_pred=y_test_pred,T_test=y_test,model_name=name,fold=split)
    return mse,rmse,r2,mae


# Define the models:
models = {
    #PyTorch-based MLP for GPU acceleration
    "Linear Regression": LinearRegression(),
    "Multilayer Perceptron": torch.nn.Sequential(
        torch.nn.Linear(276, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1)),
    "Support Vector Regression": SVR(kernel="poly"),
    "Gaussian Process Regression": GaussianProcessRegressor(kernel=RBF(), random_state=0),
    "Kernel Ridge Regression": KernelRidge(alpha=1.0, kernel='rbf'),  # Add KRR
}

# Create output directory if it doesn't exist
output_dir = 'my-regression/output_ml'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name, model in models.items():
    mse_list =[]
    rmse_list = []
    r2_list=[]
    mae_list=[]
    for fold in range(5):
        if name == "Multilayer Perceptron":
            mse,rmse,r2,mae = train(name,model,fold)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            mae_list.append(mae)
        else:
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
        

