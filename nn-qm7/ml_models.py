import os
import pickle
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


# Ensure PyTorch is installed for MLP and GPU support
try:
    import torch
    from torch.nn import Module
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
except ImportError:
    print("PyTorch not found. Install with 'pip install torch'")
    sys.exit(1)
    
# Gaussian Processes on GPU is more nuanced and library-dependent, so we'll keep it on CPU for now.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Parameters
seed = 3453
split = 4
mb = 25
hist = 0.1

# Load data (unchanged)
np.random.seed(seed)
if not os.path.exists('qm7.mat'):
    os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('qm7.mat')

# Extract training data (unchanged)
indices = [i for i in range(0, split)] + [i for i in range(split + 1, 5)]
P = dataset['P'][indices].flatten()
X = dataset['X'][P]
T = dataset['T'][0, P]


# Reshape X and split into training and testing sets (unchanged)
num_samples = X.shape[0]
X_reshaped = X.reshape(num_samples, -1)


X_train, X_test, T_train, T_test = train_test_split(X_reshaped, T, test_size=0.2)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)



# Convert data to PyTorch tensors and move to GPU if available
X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
T_train_tensor = torch.FloatTensor(T_train).to(DEVICE)
T_test_tensor = torch.FloatTensor(T_test).to(DEVICE)

# Define the kernel (RBF kernel)
kernel = 1.0 * RBF(length_scale=1.0)
 
# Create a Gaussian Process Regressor with the defined kernel
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# Define and fit models (Linear Regression and SVR are unchanged)
models = {
    # PyTorch-based MLP for GPU acceleration
    "Linear Regression": LinearRegression(),
    "Multilayer Perceptron": torch.nn.Sequential(
        torch.nn.Linear(X_reshaped.shape[1], 529),
        torch.nn.ReLU(),
        torch.nn.Linear(529, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    ).to(DEVICE),
    "Support Vector Regression": SVR(kernel="poly"),
    "Gaussian Process Regression": GaussianProcessRegressor(kernel=kernel, random_state=0),
    "Kernel Ridge Regression": KernelRidge(alpha=1.0, kernel='rbf'),  # Add KRR

}

# Create output directory if it doesn't exist
output_dir = 'output_ml'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name, model in models.items():
    if name == "Multilayer Perceptron":
        train_dataset = TensorDataset(X_train_tensor, T_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=mb, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, T_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=mb, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
        criterion = torch.nn.MSELoss()

        for epoch in range(50):  # Adjust max_iter here
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            model.eval()
            print(f'Epoch {epoch+1}, Train Loss: {loss}')
            val_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
            val_loss /= len(test_loader)
            # print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        with torch.no_grad():
            T_pred = model(X_test_tensor).cpu().numpy()
    else:
        model.fit(X_train, T_train)
        T_pred = model.predict(X_test)
    
    mse = mean_squared_error(T_test, T_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(T_test, T_pred)
    mae = mean_absolute_error(T_test, T_pred)

    print(f"\n{name}:")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  Root Mean Squared Error: {rmse:.4f}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")

    # Plotting parity plot
    plt.figure(figsize=(10, 6))
    plt.scatter(T_test, T_pred, color='green', label='Predicted vs Actual')
    plt.plot([min(T_test), max(T_test)], [min(T_test), max(T_test)], 'r--', label='Perfect Prediction')
    plt.title(f'{name} - Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f'{name.replace(" ", "_")}_parity.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()
