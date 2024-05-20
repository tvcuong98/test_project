import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1,gradient_descent=False,device="cpu"):
        super(LinearRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device=device
        # Initialize weights and bias as Parameters
        self.gradient_descent = gradient_descent
        self.weights = torch.randn(input_dim, self.output_dim, requires_grad=False).to(device)
        print(self.weights.device)
        self.bias = torch.randn(self.output_dim, requires_grad=False).to(device)
        # Concatenate directly (no need to convert in forward)
        self.w = torch.cat([self.bias.view(-1,1),self.weights],dim=0).to(device)
    def forward(self,x):
        x = x.to(self.device)
        x = torch.cat([torch.ones(x.shape[0], 1).to(self.device), x], dim=1)
        result = torch.einsum('bi, io -> bo', x, self.w)
        return result
    def fit(self,x,y):
        if (self.gradient_descent==False):
            if type(x)==np.ndarray and type(y)==np.ndarray :
                x = torch.from_numpy(x).float().to(self.device)
                y = torch.from_numpy(y).float().to(self.device)
            else: #x,y are tensors
                x = x.float().to(self.device)
                y = y.float().to(self.device)

            # Prepend a column of ones to X (for the intercept term)
            x = torch.cat([torch.ones(x.shape[0], 1).to(self.device), x], dim=1)

            # Efficiently calculate the Moore-Penrose pseudoinverse
            x_plus = torch.linalg.pinv(x) # x_plus is (X_t*X)^-1*X

            # Compute the weights (w = X_plus * y)
            self.w = x_plus @ y 
class RidgeRegression(nn.Module):
    """
    Solution to the ridge regression algorithm is in here https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html
    """
    def __init__(self, input_dim, output_dim=1,gradient_descent=False,device="cpu"):
        super(RidgeRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device=device
        # Initialize weights and bias as Parameters
        self.gradient_descent = gradient_descent
        self.weights = torch.randn(input_dim, self.output_dim, requires_grad=False).to(device)
        self.bias = torch.randn(self.output_dim, requires_grad=False).to(device)
        # Concatenate directly (no need to convert in forward)
        self.w = torch.cat([self.bias.view(-1,1),self.weights],dim=0).to(device)
    def forward(self,x):
        x = x.to(self.device)
        x = torch.cat([torch.ones(x.shape[0], 1).to(self.device), x], dim=1)
        result = torch.einsum('bi, io -> bo', x, self.w)
        return result
    def fit(self,x,y):
        if (self.gradient_descent==False):
            if type(x)==np.ndarray and type(y)==np.ndarray :
                x = torch.from_numpy(x).float().to(self.device)
                y = torch.from_numpy(y).float().to(self.device)
            else: #x,y are tensors
                x = x.float().to(self.device)
                y = y.float().to(self.device)
            # Prepend a column of ones to X (for the intercept term)
            x = torch.cat([torch.ones(x.shape[0], 1).to(self.device), x], dim=1)
            lambda_reg = 0.1  # Regularization parameter
            x= x.permute(-1,-2) # x is now dim*batch_size 
            # Compute XX^T 
            XXT = torch.mm(x, x.t())  # x is now dim*batch_size  -> x_t is batch_size * dim
                                    # XXT must be dim * dim in the end

            # Add lambda * I to XX^T
            lambda_I = lambda_reg * torch.eye(XXT.size(0)).to(self.device)
            A = XXT + lambda_I

            # Compute the inverse of A
            A_inv = torch.inverse(A)  # A_inv is dim * dim in the end

            # Compute Xy^T

            XyT = torch.mm(x, y.view(-1, 1)) # dim*batch_size multiply with batch_size_dim

            # Compute the weight vector w
            self.w = torch.mm(A_inv, XyT)
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out
def kernel_regression(X_train, y_train, X_test, bandwidth=1.0, kernel_func='gaussian'):
    """Performs kernel regression.

    Args:
        X_train: A 2D tensor of shape (num_train_samples, num_features), representing the training features.
        y_train: A 1D tensor of shape (num_train_samples,), representing the training targets.
        X_test: A 2D tensor of shape (num_test_samples, num_features), representing the test features.
        bandwidth: A scalar bandwidth parameter for the kernel function.
        kernel_func: A string indicating the kernel function to use ('gaussian' is the default).

    Returns:
        A 1D tensor of shape (num_test_samples,) containing the predicted values for the test points.
    """

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Ensure that the number of features matches between training and test data
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Number of features must match between training and test data")

    # Calculate pairwise distances (automatically handles n-dimensional input)
    dist = torch.cdist(X_test, X_train)

    if kernel_func == 'gaussian':
        # Gaussian kernel
        weights = torch.exp(-dist**2 / (2 * bandwidth**2))
    elif kernel_func == 'laplacian':  # Added Laplacian kernel
            weights = torch.exp(-dist / bandwidth)
    elif kernel_func == 'linear':   # Added Linear kernel
            weights = torch.clamp(1 - dist / bandwidth, min=0)  # Ensure non-negativity
    elif kernel_func == 'polynomial':
            degree = 3  # You can customize the degree
            weights = (1 + torch.matmul(X_test, X_train.T) / bandwidth) ** degree
    else:
        raise ValueError("Unsupported kernel function")

    # Normalize weights along the training sample dimension
    weights /= _norm_no_nan(weights,axis=1,keepdims=True)
    # Matrix multiplication to get predictions
    predictions = weights @ y_train

    return predictions