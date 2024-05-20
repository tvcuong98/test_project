import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_scatter
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = nn.Linear(self.in_channels, self.out_channels) # linear transformation that you apply to embedding  for central node.
             
        self.lin_r = nn.Linear(self.in_channels, self.out_channels) # linear transformation that you apply to aggregated(already) info from neighbors.

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()      

    def forward(self, data, size = None):
        x = data.x
        edge_index = data.edge_index
        prop = self.propagate(edge_index, x=(x, x), size=size) # see Messsage.Passing.propagate() in https://pytorch-geometric.readthedocs.io/en/latest/
        out = self.lin_l(x) + self.lin_r(prop)
        if self.normalize:
          out = F.normalize(out, p=2)
        data.x = out
        
        return data
    
    # Implement your message function here.
    def message(self, x_j):
      out = x_j
      return out
    
    # Implement your aggregate function here.
    def aggregate(self, inputs, index, dim_size = None):
        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        # since 
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean') # see https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
        return out
class GraphSageRegression(nn.Module):
    def __init__(self, in_channels,out_channels,hid_channels=64,normalize = True):
        
        super().__init__()
         
        self.gnn = nn.Sequential(
            GraphSage(in_channels, hid_channels, normalize),
            GraphSage(hid_channels, hid_channels, normalize),
            GraphSage(hid_channels, hid_channels, normalize),
            GraphSage(hid_channels, out_channels, normalize)
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