import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeInference(nn.Module):
    def __init__(self, input_dim):
        super(EdgeInference, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, node_embeddings):
        # node_embeddings: tensor of shape (num_nodes, input_dim)
        edges = self.linear(node_embeddings)  # Shape: (num_nodes, 1)
        edges = torch.sigmoid(edges)  # Shape: (num_nodes, 1)
        return edges
class GNN(nn.Module):
    
    def __init__(self, in_channels, n_classes, t_size, latent, edge_importance_weighting=True, dataset='ntu', **kwargs):
        super().__init__()