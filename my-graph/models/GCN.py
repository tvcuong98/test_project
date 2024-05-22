import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.pool import global_mean_pool,global_max_pool
from torch_sparse import SparseTensor
import torch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add') # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, data):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]
        x = data.x
        edge_index = data.edge_index
        ### DO NOT ADD SELF_LOOP!!!! SELF Loop has adready been add in QM7
        # Step 1: Add self-loops to the adjacency matrix. 
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        data.x = self.propagate(edge_index, x=x)
        # Step 3-5: Start propagating messages.
        return data

    def message(self, x_j, edge_index, size):
        # x_j has shape [num_edges, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
class GCNRegression(nn.Module):
    def __init__(self, in_channels,out_channels,hid_channels=64):
        
        super().__init__()
         
        self.gnn = nn.Sequential(
            GCNConv(in_channels, hid_channels),
            GCNConv(hid_channels, hid_channels),
            GCNConv(hid_channels, hid_channels),
            GCNConv(hid_channels, out_channels)
        )
        self.linear=nn.Linear(out_channels,1)
 
         
    def forward(self, data):      
        '''
        ''' 
        x_gcn = self.gnn(data)
        x_dense,mask=to_dense_batch(x_gcn.x,data.batch) # it is now batch, nodes, out_channels
        x_new = x_dense.mean(dim=1) # it is now batch, out_channels
        out = self.linear(x_new)
        return out