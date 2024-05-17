
import torch
import numpy as np
import math
from utils import *
"""
About all the types of matrix multiplications
3. The np.dot() Function
For 2D arrays, np.dot() behaves the same as @ and np.matmul().
For 1D arrays (vectors), it performs the dot product.
For higher-dimensional arrays, it performs a sum-product over the last axis of the first array and the second-to-last axis of the second array.
4.Element-wise Multiplication: 
If you want to multiply corresponding elements of two arrays, use the * operator or the np.multiply() function. This is NOT matrix multiplication.
"""
class LinearRegression(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.linear = None  # Placeholder for the linear layer

    def fit(self, X_train, y_train):
        # Convert data to PyTorch tensors and move to the specified device (GPU if available)
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        
        # Add a bias term (column of ones)
        Xhat = torch.cat((torch.ones((X_train.shape[0], 1), device=self.device), X_train), dim=1)

        # Initialize the linear layer dynamically based on input size
        self.linear = torch.nn.Linear(Xhat.shape[1], 1, device=self.device)
        
        # Use closed-form solution for linear regression
        A = Xhat.T @ Xhat
        b = Xhat.T @ y_train
        self.linear.weight = torch.nn.Parameter(torch.linalg.pinv(A) @ b)

    def forward(self, X):
        # Convert input to PyTorch tensor and move to device
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Add bias term
        Xhat = torch.cat((torch.ones((X.shape[0], 1), device=self.device), X), dim=1)

        # Predict using the linear layer
        y_predict = self.linear(Xhat)

        # Return prediction
        return y_predict
# Ridge Regression:

   
# Lasso Regression
class LassoRegression():
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
 
    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self
 
    # Helper function to update weights in gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X)
        # calculate gradients
        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) +
                         self.l1_penalty) / self.m
            else:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) -
                         self.l1_penalty) / self.m
 
        db = -2 * np.sum(self.Y - Y_pred) / self.m
 
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self
 
    # Hypothetical function h(x)
    def predict(self, X):
        return X.dot(self.W) + self.b
class RidgeRegression() : 
      
    def __init__( self, learning_rate, iterations, l2_penality ) : 
          
        self.learning_rate = learning_rate         
        self.iterations = iterations         
        self.l2_penality = l2_penality 
          
    # Function for model training             
    def fit( self, X, Y ) : 
          
        # no_of_training_examples, no_of_features         
        self.m, self.n = X.shape 
          
        # weight initialization         
        self.W = np.zeros( self.n ) 
          
        self.b = 0        
        self.X = X         
        self.Y = Y 
          
        # gradient descent learning 
                  
        for i in range( self.iterations ) :             
            self.update_weights()             
        return self
      
    # Helper function to update weights in gradient descent 
      
    def update_weights( self ) :            
        Y_pred = self.predict( self.X ) 
          
        # calculate gradients       
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +               
               ( 2 * self.l2_penality * self.W ) ) / self.m      
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
        # update weights     
        self.W = self.W - self.learning_rate * dW     
        self.b = self.b - self.learning_rate * db         
        return self
      
    # Hypothetical function  h( x )  
    def predict( self, X ) :     
        return X.dot( self.W ) + self.b 
from scipy.stats import chi2, multivariate_normal
class BayesianRegression(object):
    """Bayesian regression model. If poly_degree is specified the features will
    be transformed to with a polynomial basis function, which allows for polynomial
    regression. Assumes Normal prior and likelihood for the weights and scaled inverse
    chi-squared prior and likelihood for the variance of the weights.

    Parameters:
    -----------
    n_draws: float
        The number of simulated draws from the posterior of the parameters.
    mu0: array
        The mean values of the prior Normal distribution of the parameters.
    omega0: array
        The precision matrix of the prior Normal distribution of the parameters.
    nu0: float
        The degrees of freedom of the prior scaled inverse chi squared distribution.
    sigma_sq0: float
        The scale parameter of the prior scaled inverse chi squared distribution.
    poly_degree: int
        The polynomial degree that the features should be transformed to. Allows
        for polynomial regression.
    cred_int: float
        The credible interval (ETI in this impl.). 95 => 95% credible interval of the posterior
        of the parameters.

    Reference:
        https://github.com/mattiasvillani/BayesLearnCourse/raw/master/Slides/BayesLearnL5.pdf
    """
    def __init__(self, n_draws, mu0, omega0, nu0, sigma_sq0, poly_degree=0, cred_int=95):
        self.w = None
        self.n_draws = n_draws
        self.poly_degree = poly_degree
        self.cred_int = cred_int

        # Prior parameters
        self.mu0 = mu0
        self.omega0 = omega0
        self.nu0 = nu0
        self.sigma_sq0 = sigma_sq0

    # Allows for simulation from the scaled inverse chi squared
    # distribution. Assumes the variance is distributed according to
    # this distribution.
    # Reference:
    #   https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution
    def _draw_scaled_inv_chi_sq(self, n, df, scale):
        X = chi2.rvs(size=n, df=df)
        sigma_sq = df * scale / X
        return sigma_sq

    def fit(self, X, y):

        # If polynomial transformation
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        n_samples, n_features = np.shape(X)

        X_X = X.T.dot(X)

        # Least squares approximate of beta
        beta_hat = np.linalg.pinv(X_X).dot(X.T).dot(y)

        # The posterior parameters can be determined analytically since we assume
        # conjugate priors for the likelihoods.

        # Normal prior / likelihood => Normal posterior
        mu_n = np.linalg.pinv(X_X + self.omega0).dot(X_X.dot(beta_hat)+self.omega0.dot(self.mu0))
        omega_n = X_X + self.omega0
        # Scaled inverse chi-squared prior / likelihood => Scaled inverse chi-squared posterior
        nu_n = self.nu0 + n_samples
        sigma_sq_n = (1.0/nu_n)*(self.nu0*self.sigma_sq0 + \
            (y.T.dot(y) + self.mu0.T.dot(self.omega0).dot(self.mu0) - mu_n.T.dot(omega_n.dot(mu_n))))

        # Simulate parameter values for n_draws
        beta_draws = np.empty((self.n_draws, n_features))
        for i in range(self.n_draws):
            sigma_sq = self._draw_scaled_inv_chi_sq(n=1, df=nu_n, scale=sigma_sq_n)
            beta = multivariate_normal.rvs(size=1, mean=mu_n[:,0], cov=sigma_sq*np.linalg.pinv(omega_n))
            # Save parameter draws
            beta_draws[i, :] = beta

        # Select the mean of the simulated variables as the ones used to make predictions
        self.w = np.mean(beta_draws, axis=0)

        # Lower and upper boundary of the credible interval
        l_eti = 50 - self.cred_int/2
        u_eti = 50 + self.cred_int/2
        self.eti = np.array([[np.percentile(beta_draws[:,i], q=l_eti), np.percentile(beta_draws[:,i], q=u_eti)] \
                                for i in range(n_features)])

    def predict(self, X, eti=False):

        # If polynomial transformation
        if self.poly_degree:
            X = polynomial_features(X, degree=self.poly_degree)

        y_pred = X.dot(self.w)
        # If the lower and upper boundaries for the 95%
        # equal tail interval should be returned
        if eti:
            lower_w = self.eti[:, 0]
            upper_w = self.eti[:, 1]
            y_lower_pred = X.dot(lower_w)
            y_upper_pred = X.dot(upper_w)
            return y_pred, y_lower_pred, y_upper_pred

        return y_pred
