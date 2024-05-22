from typing import Callable, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class QM7b(InMemoryDataset):
    r"""The QM7b dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    7,211 molecules with 14 regression targets.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 7,211
          - ~15.4
          - ~245.0
          - 0
          - 14
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'qm7b.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        from scipy.io import loadmat

        data = loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])
        target = torch.from_numpy(data['T']).to(torch.float) #batch,num_classes

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero(
                as_tuple=False).t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y = target[i].view(1, -1) #batch,num_classes -> 1,num_class -> 1,num_class (the .view() did do something)
            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

class QM7(InMemoryDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        train=True,
        split=-1
    ) -> None:
        self.split = split
        self.train = train
        assert isinstance(split, int) and -1 <= split <= 4, 'Fold must be in range [-1, 4]'
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        # self.load(self.processed_paths[0]) # equivalent, but the latter is easier to understand
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self) -> str:
        return 'qm7.mat'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.train}_{self.split}.pt'

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None: # I think this guy is called inside the init
        from scipy.io import loadmat
        
        # load the data, create the suitable mask
        data = loadmat(self.raw_paths[0])
        fold_splits = torch.from_numpy(data['P']) # (5,1433)
        fold_split = fold_splits[self.split] # (1433,)
        mask = torch.zeros(len(data['X']), dtype=bool) # (7165,), all False
        if (self.train==False):
            mask[fold_split]=True # turn on True for the sample that have its index listed in split
        elif (self.train==True):
            mask[fold_split]=True # turn on True for the sample that have its index listed in split
            mask = ~mask

        xyz_matrix = torch.from_numpy(data['R'][mask])  # samples,23,3 
        charge_matrix = torch.from_numpy(data['Z'][mask])  # samples,23
        coulomb_matrix = torch.from_numpy(data['X'][mask]) # samples,23,23
        target = torch.from_numpy(data['T'][:,mask]).to(torch.float) # 1,samples
        target = target.permute(1,0) # samples,1

        data_list = []
        for i in range(target.shape[0]):
            node_attr_xyz=xyz_matrix[i] #23,3
            node_attr_charge=charge_matrix[i].view(-1,1) #23,1
            concatenated_node_attr = torch.cat((node_attr_xyz,node_attr_charge),-1)

            edge_index = coulomb_matrix[i].nonzero(
                as_tuple=False).t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y = target[i].view(1, -1) # samples,1 -> 1,1 <batch,num_class> -> 1,1 (yes, the .view() must have its purpose)
            data = Data(x=concatenated_node_attr,edge_index=edge_index, edge_attr=edge_attr, y=y)
            # data.num_nodes = edge_index.max().item() + 1
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        # self.save(data_list, self.processed_paths[0])        
        data,slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
import scipy
from torch.utils.data import Dataset
import numpy as np
class QM7_pytorch_dataset(Dataset):
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
            self.Z = data['Z'][mask]
        else:
            self.X = data['X']
            self.y = data.T['T']
            self.R = data['R']
            self.Z = data['Z']
        self.feature = np.concatenate((self.R, np.expand_dims(self.Z, axis=-1)), axis=-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.y[idx]
        feature = torch.tensor(feature)
        label = torch.tensor(label)
        return feature, label