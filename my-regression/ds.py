import scipy.io
from torch.utils.data import Dataset
import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
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
    
class QM7_Deluxe(Dataset):
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
        num_atoms = 23
        iu = np.triu_indices(num_atoms,k=0) # default: Includes the main diagonal and everything above.
        iu_dist = np.triu_indices(num_atoms,k=1) # for the pairwise distance matrix, all diagonol entries will be 0 
        # What the np.triu_indices return?
        # The first array contains the row indices of the upper triangle elements.
        # The second array contains the corresponding column indices.

        CM = np.zeros((self.X.shape[0], num_atoms*(num_atoms+1)//2), dtype=float) # shape 7165*276 (276 is the total number of upper triangular element in the 23 x 23 matrix)
        eigs = np.zeros((self.X.shape[0], num_atoms), dtype=float)                # shape 7165*23
        centralities = np.zeros((self.X.shape[0], num_atoms), dtype=float)        # shape 7165*23
        interatomic_dist = np.zeros((self.X.shape[0], ((num_atoms*num_atoms)-num_atoms)//2), dtype=float)  # shape 7165*253 (276 is the total number of upper triangular element in the 23 x 23 matrix, excluding the diagonals (276-23))

        verbose=True

        for i, cm in enumerate(self.X): # loop through each sample
            coulomb_vector = cm[iu]  # choose only the upper triangular elements : 276
            # Sort elements by decreasing order
            shuffle = np.argsort(-coulomb_vector)
            CM[i] = coulomb_vector[shuffle] # the triu elements , sorted decressing: 276 : hold the in4 about the couloub value between 23 nodes (removed the redundant in4)
            dist = squareform(pdist(self.R[i])) # shape 23 * 23 , contain the pair-wise distance between the 23 nodes
            # we can extract the upper triangle of the distance matri: return vector of dimension (1,num_atoms)
            dist_vector = dist[iu_dist] # shape 253, since distance matrix are symmetric -> take only the upper triangular (276) -> since self-distance is meaningless, remove the diagonal -> 253
            shuffle = np.argsort(-dist_vector) 
            interatomic_dist[i] = dist_vector[shuffle] # still shape 253, holds the information about pair-wise distance of 23 nodes (removed the redundant in4)
            
            w,v = np.linalg.eig((dist)) # Eigenvalues (w): (23,) Represent how much a vector is stretched or shrunk when transformed by the distance matrix
                                        # Eigenvectors (v): (23,23) These are vectors that, when multiplied by the distance matrix, only change in scale (stretched or shrunk) but not in direction
            eigs[i] = w[np.argsort(-w)]
            centralities[i] = np.array(list(nx.eigenvector_centrality(nx.Graph(dist)).values())) # (23,) , value from 0 -> 1
            """Explaination of centrailities
            Captures the relative importance of each atom in the molecule's connectivity structure. Atoms with high eigenvector centrality might:

            + Be central to the overall molecular shape.
            + Play crucial roles in chemical reactions or interactions.
            + Be important for understanding the molecule's properties and behavior.
            """
            # if verbose and i % 500 == 0:
            #     print("Processed {} molecules".format(i))
    
        self.X = np.concatenate((CM, eigs, centralities, interatomic_dist), axis=1) # (7165, 575) , where 575 = 276 + 23 + 23 + 253

        
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)