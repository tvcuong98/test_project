import scipy.io
from torch.utils.data import Dataset
import torch
import numpy as np

def transform(X,Y,type="hard_coded"):
    """"""
    if type=="hard_coded": # normalized both the feature and the label
            x_mean, x_std = 1.6812959, 6.700323
            y_mean, y_std = -1538.0377, 223.91891


            X = (torch.Tensor(X) - x_mean) / x_std
            Y = (torch.Tensor(Y) - y_mean) / y_std
    return X,Y
class QM7(Dataset):
    def __init__(self, path, train=True, split=-1):
        assert isinstance(split, int) and -1 <= split <= 4, 'Fold must be in range [-1, 4]'
        super().__init__()
        self.path = path
        data = scipy.io.loadmat(path)
        if split != -1:
            split = data['P'][split]
            mask = torch.zeros(data['T'].size, dtype=bool)
            mask[split] = True
            if train:
                mask = ~mask
            print(mask.shape)
            self.X = data['X'][mask]
            self.y = data['T'].T[mask]
            self.R = data['R'][mask]
        else:
            self.X = data['X']
            self.y = data.T['T']
            self.R = data['R']


        # Extracting upper triangular indexes from 2D matrix: Use to dimensionality reduction when the pairwise features are symmetric, which they are 
        # at this point the X.shape is still Bx23x23 
        X_proc = []
        for x in self.X:
            X_proc.append(x[np.triu_indices(23)])
        self.X = np.vstack(X_proc)
        # now the X.shape is still Bx276 (23x23=529 != 276)
        # self.X,self.y = transform(self.X,self.y,"hard_coded")
        """
        In-depth explaination:
        

        Purpose: Extracting Upper Triangular Elements

        Transform the 2D feature matrix X (likely representing pairwise features between atoms or nodes in a graph) into a 1D array
        by keeping only the elements from the upper triangular part of each 23x23 matrix within X. Here's how it works:

        ### Iteration:

        It iterates through each 23x23 matrix x within your self.X array.
        np.triu_indices(23):

        This NumPy function generates the indices for the upper triangular part of a 23x23 matrix.
        For example, for a 3x3 matrix, the upper triangular indices would be (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2).
        ## Indexing:

        It uses the generated indices to select the elements from the upper triangular part of the x matrix.
        ## Appending:

        The extracted upper triangular elements are appended to the X_proc list.
        ## Stacking:

        Finally, the np.vstack function vertically stacks all the extracted upper triangular elements into a 1D array, which becomes the new value of self.X.
        """

        
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)