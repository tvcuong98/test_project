import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.pool import global_mean_pool,global_max_pool
from torch_sparse import SparseTensor
# calculating the degree of a node is calculating the number of neighbor it have
# """
# Add self-loops to the adjacency matrix.

# Linearly transform node feature matrix.

# Compute normalization coefficients.

# Normalize node features in "MESSAGE "

# Sum up neighboring node features ("add" aggregation).

# Apply a final bias vector.
# """
# class GCNConv(MessagePassing):
#     def __init__(self,in_dims,out_dims,hidden_dims,final=False):

#         """
#         """
#         super().__init__(aggr='add') # "Add" aggregation (Step 5).
#         self.linear1 = Linear(in_dims,hidden_dims,bias=False) #(in_dims x out_dims)
#         self.linear2 = Linear(hidden_dims,out_dims,bias=False) #(in_dims x out_dims)
#         self.bias = Parameter(torch.empty(out_dims)) #out_dims
#         self.final = final
#         self.reset_parameters()
#     def reset_parameters(self):
#         self.linear1.reset_parameters()
#         self.linear2.reset_parameters()
#         self.bias.data.zero_()
#     def forward(self,data):
#         r"""
#         x has shape [N, in_channels]
#         edge index have shape [2,E]
#         """
#         # Step 1 : Add self-loops to adjacentcy matrix (no need, we have already did it)
#         # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#         x= data.x
#         edge_index = data.edge_index

#         # Step 2: Linearly transform node feature matrix with W
#         x = self.linear1(x)
#         # Step 3 : Compute the normalization:
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         out = self.propagate(edge_index, x=x, norm=norm) # I think at this step the edge_index is transformed to adj matrix
#                                                          # then automatically multiply the adj matrix with x


#         # all over again
#                 # Step 2: Linearly transform node feature matrix with W
#         out = self.linear2(x)
#         # Step 3 : Compute the normalization:
#         row, col = edge_index
#         deg = degree(col, out.size(0), dtype=out.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         out = self.propagate(edge_index, x=out, norm=norm) # I think at this step the edge_index is transformed to adj matrix
#                                                          # then automatically multiply the adj matrix with x

#         # Step 6: Apply a final bias vector.
#         out = out + self.bias



#         if self.final ==True:
#             out = global_mean_pool(out, data.batch) # global mean pool across all the node in each sample of a batch 
#                                                     # -> out is shaped (batch_size*num_nodes,out_dims) -> (batch_size,out_dims)
#         return out

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]
#         result = norm.view(-1, 1) * x_j
#         # Step 4: Normalize node features.
#         return result
# """
import torch
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

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, x=x)

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
