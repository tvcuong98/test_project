import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
class GAT(torch.nn.Module):
    def __init__(self,in_channels,out_channels,hid_channels=64,in_head=8,hid_head=1,out_head=1):
        super(GAT, self).__init__()
        
        self.gat1 = GATConv(in_channels, hid_channels, heads=in_head,dropout=0.6)
        self.gat2 = GATConv(hid_channels*in_head, hid_channels*in_head,heads=hid_head,dropout=0.6)
        self.gat3 = GATConv(hid_channels*in_head, hid_channels*in_head,heads=hid_head,dropout=0.6)
        self.gat4 = GATConv(hid_channels*in_head, out_channels*out_head, heads=out_head,dropout=0.6)
        self.linear=nn.Linear(out_channels,1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv1(x, edge_index)
        # x = F.elu(x)
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)
        
        # return F.log_softmax(x, dim=1)
        x = self.gat1(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat3(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x_gat = self.gat4(x,edge_index)
        x_dense,mask=to_dense_batch(x_gat,data.batch) # it is now batch, nodes, out_channels
        x_new = x_dense.mean(dim=1) # it is now batch, out_channels
        out = self.linear(x_new)
        return out