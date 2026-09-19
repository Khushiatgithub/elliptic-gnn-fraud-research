"""
Graph Convolutional Network (GCN) Architecture for Illicit Transaction Detection.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(nn.Module):
    """
    Graph Convolutional Network with configurable layers, hidden channels, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        else:
            # First layer
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            # Intermediate layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            # Final GNN layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        logits = self.classifier(x).squeeze(-1)
        return logits
