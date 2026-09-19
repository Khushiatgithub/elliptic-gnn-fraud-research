"""
Graph Attention Network (GAT) Architecture for Illicit Transaction Detection.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    """
    Graph Attention Network with multi-head attention and configurable depth.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # Determine head dimension so total feature width per layer matches hidden_channels
        head_dim = max(1, hidden_channels // heads)
        mid_dim = head_dim * heads
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=1, concat=False, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        else:
            # First layer: multi-head with concat
            self.convs.append(GATConv(in_channels, head_dim, heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(mid_dim))
            
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(mid_dim, head_dim, heads=heads, concat=True, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(mid_dim))
                
            # Final GNN layer (single head average / projection)
            self.convs.append(GATConv(mid_dim, hidden_channels, heads=1, concat=False, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        # Final classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unscaled logits of shape [N].
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        logits = self.classifier(x).squeeze(-1)
        return logits
