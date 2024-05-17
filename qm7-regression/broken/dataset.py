import torch
import scipy.io
from torch_geometric.data import Data, InMemoryDataset, download_url


# Ended up not using this because I couldn't get GCN to work.
class QM7(InMemoryDataset):
    """
    torch_geometric `InMemoryDataset` for QM7.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, fold=-1, train=True):
        assert isinstance(fold, int) and -1 <= fold <= 4, 'Fold must be in range [-1, 4]'
        self.fold = fold
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['qm7.mat']

    @property
    def processed_file_names(self):
        return [f'qm7_fold{self.fold}.{self.train}.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url('http://quantum-machine.org/data/qm7.mat', self.raw_dir)
    
    def _adj_to_data(self, X):
        E = []
        E_attr = []
        nnodes = X.shape[0]
        for u in range(nnodes):
            for v in range(nnodes):
                if X[u, v] != 0:
                    assert X[u, v] > 0
                    E.append((u, v))
                    E_attr.append(X[u, v])
                    E.append((v, u))
                    E_attr.append(X[u, v])
        return torch.Tensor(E).long().T, torch.Tensor(E_attr)[..., None]

    def process(self):
        # Read data into a huge `Data` list
        data_list = []
        mat = scipy.io.loadmat(self.raw_paths[0])
        X = mat['X']
        y = mat['T'].T.reshape(-1)
        R = mat['R']

        X_max = X.max()
        X = X / X_max

        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std

        R_mean = R.mean()
        R_std = R.std()
        R = (R - R_mean) / R_std

        self.X_max = X_max
        self.y_mean = y_mean
        self.y_std = y_std
        self.R_mean = R_mean
        self.R_std = R_std
        
        
        if self.fold != -1:
            split = mat['P'][self.fold]
            mask = torch.zeros(y.size, dtype=bool)
            mask[split] = True
            if self.train: mask = ~mask
            self.X = X[mask]
            self.y = y[mask]
            self.R = R[mask]
        else:
            self.X = X
            self.y = y
            self.R = R
        
        for graph, target, pos in zip(X, y, R):
            edge_data = self._adj_to_data(graph)
            data_list.append(Data(x=torch.ones((23, 1)), edge_index=edge_data[0], edge_attr=edge_data[1], y=torch.Tensor([target])))
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = QM7(root='./qm7', fold=0, train=True)
    print(dataset[2])