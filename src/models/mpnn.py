import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.dimenet import BesselBasisLayer

from src.models import modules

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric as tg

class MPNN(nn.Module):
    """Message passing model for graph-level predictions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        num_types=10,
        num_basis=12,
        cutoff=4.0,
        dim=64,
        out_dim=1,
        depth=5,
        message_depth=1,
        update_depth=1,
        head_depth=1,
        norm=None,
        act=nn.SiLU,
        aggr="add",
        pooler=tg.nn.global_add_pool,
        **kwargs
    ):
        super().__init__()

        message_dims = [dim * 2 + num_basis] + [dim] * (message_depth + 1)
        update_dims = [2 * dim] + [dim] * (update_depth + 1)
        head_dims = [dim] * (head_depth + 1) + [out_dim]
        self.embedding = nn.Embedding(num_types, dim)
        self.basis = BesselBasisLayer(num_radial=num_basis, cutoff=cutoff)

        self.layers = nn.ModuleList()
        for i in range(depth):
            message_net = modules.build_mlp(message_dims, act, norm)
            update_net = modules.build_mlp(update_dims, act, norm)
            self.layers.append(MPNNLayer(message_net, update_net, aggr))

        self.pooler = pooler
        self.head = modules.build_mlp(head_dims, act, norm)

    def distance(self, pos, edge_index):
        row, col = edge_index
        dist = pos[row] - pos[col]
        return dist.norm(p=2, dim=-1)

    def forward(self, graph, logger=None):
        z = graph.z
        edge_index = graph.edge_index
        batch = graph.batch
        pos = graph.pos

        x = self.embedding(z)
        dist = self.distance(pos, edge_index)
        dist = self.basis(dist)

        for layer in self.layers:
            x = x + layer(x, edge_index, dist)
        x = self.pooler(x, batch)
        x = self.head(x)

        return x


class MPNNLayer(MessagePassing):
    """Message Passing Layer"""

    def __init__(self, message_net, update_net, aggr):
        super().__init__(aggr=aggr)
        self.message_net = message_net
        self.update_net = update_net

    def forward(self, x, edge_index, edge_attr=None):
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_i, x_j, edge_attr):
        x = [x_i, x_j, edge_attr]
        x = [_ for _ in x if _ is not None]
        x = torch.cat(x, dim=-1)
        return self.message_net(x)

    def update(self, message, x):
        x = torch.cat([x, message], dim=-1)
        return self.update_net(x)

    def __repr__(self):
        return "{}(\n message_net={} \n update_net={})".format(
            self.__class__.__name__, self.message_net, self.update_net
        )