
import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.pool import global_mean_pool,global_max_pool
from torch_geometric.utils import to_dense_batch
# calculating the degree of a node is calculating the number of neighbor it have
# """
# Add self-loops to the adjacency matrix.

# Linearly transform node feature matrix.

# Compute normalization coefficients.

# Normalize node features in "MESSAGE "

# Sum up neighboring node features ("add" aggregation).

# Apply a final bias vector.
# """
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels,use_edge_attr=False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.use_edge_attr=use_edge_attr
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, data):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x,edge_index,edge_attr = data.x,data.edge_index,data.edge_attr
        
        ### DO NOT ADD SELF_LOOP!!!! SELF Loop has adready been add in QM7
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.use_edge_attr: self.edge_attr = edge_attr
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias
        data.x=out
        return data

    def message(self,x_i, x_j, norm): # x_i is [E,out_channels], x_j [E,out_channels]
        r"""
        The purpose of the message is to crafting the message, meaning to calculate the feature message from OTHER NODES, working with x_j,returning x_j
        Since x_j represent the feature from OTHER NODE : x_j[E,out_channel] <- the number of edges stem from a node is also its number of neighbors !!
        """
        # Step 4: Normalize node features.
        if self.use_edge_attr: result = x_j*(self.edge_attr.view(-1,1))
        else : result = x_j
        return norm.view(-1, 1) * result
class GCNRegression(nn.Module):
    def __init__(self, in_channels,out_channels,use_edge_attr=False,hid_channels=64):
        
        super().__init__()
         
        self.gnn = nn.Sequential(
            GCNConv(in_channels, hid_channels,use_edge_attr=False),
            GCNConv(hid_channels, hid_channels,use_edge_attr=False),
            GCNConv(hid_channels, hid_channels,use_edge_attr=False),
            GCNConv(hid_channels, out_channels,use_edge_attr=False)
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
