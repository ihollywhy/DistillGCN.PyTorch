"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn

import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, GATConv, GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GCN, self).__init__()
        self.g = g
        self.gat_layers = []
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_dim, num_hidden))
        for _ in range(1, num_layers):
            self.gat_layers.append(GraphConv(num_hidden, num_hidden))
        
        self.gat_layers.append( GraphConv(num_hidden, num_classes) )
        
    def forward(self, inputs, middle=False):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h)
            middle_feats.append(h)
            h = F.relu(h)
        # output projection
        logits = self.gat_layers[-1](self.g, h)
        if middle:
            return logits, middle_feats
        return logits


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, None))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, graph, inputs, middle=False):
        h = inputs
        middle_feats = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](graph, h).flatten(1)
            middle_feats.append(h)
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](graph, h).mean(1)
        if middle:
            return logits, middle_feats
        return logits