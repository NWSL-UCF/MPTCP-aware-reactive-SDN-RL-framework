# RL2/models/gnn_models.py
"""
GNN Model Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for graph representation learning"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.2):
        super(GraphSAGEEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index, batch=None):
        # Apply convolutions
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.2, heads=4):
        super(GATEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, 
                       heads=heads, dropout=dropout)
            )
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                GATConv(hidden_dim * heads, output_dim, 
                       heads=1, concat=False, dropout=dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        # Apply attention layers
        for i, conv in enumerate(self.convs[:-1]):
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class GCNEncoder(nn.Module):
    """Basic GCN encoder (for comparison)"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.2):
        super(GCNEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x