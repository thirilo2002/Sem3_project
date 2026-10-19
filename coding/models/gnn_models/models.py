"""Three heterogeneous GNN model architectures from the supplied script."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, HGTConv, RGCNConv

from .config import HIDDEN_DIM, DROPOUT, HGT_HEADS


class HeteroSAGE(nn.Module):
    """HeteroConv + SAGEConv: relation-specific neighbor aggregation."""

    def __init__(self, metadata, in_dims, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.input_projection = nn.ModuleDict({
            nt: nn.Linear(in_dims[nt], hidden) for nt in node_types
        })

        def make_conv():
            return HeteroConv(
                {et: __import__("torch_geometric.nn", fromlist=["SAGEConv"]).SAGEConv(
                    (hidden, hidden), hidden
                ) for et in edge_types},
                aggr="sum",
            )

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, batch):
        x_dict = {
            nt: F.relu(self.input_projection[nt](x))
            for nt, x in batch.x_dict.items()
        }
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2)):
            updated = conv(x_dict, batch.edge_index_dict)
            x_dict = {
                nt: F.dropout(
                    norm[nt](F.relu(updated.get(nt, x))),
                    p=self.dropout,
                    training=self.training,
                )
                for nt, x in x_dict.items()
            }
        return self.classifier(
            x_dict["provider"][: batch["provider"].batch_size]
        )


class HeteroHGT(nn.Module):
    """HGTConv: heterogeneous multi-head attention."""

    def __init__(
        self, metadata, in_dims, hidden=HIDDEN_DIM,
        heads=HGT_HEADS, dropout=DROPOUT
    ):
        super().__init__()
        node_types, _ = metadata
        self.dropout = dropout
        self.input_projection = nn.ModuleDict({
            nt: nn.Linear(in_dims[nt], hidden) for nt in node_types
        })
        self.conv1 = HGTConv(hidden, hidden, metadata=metadata, heads=heads)
        self.conv2 = HGTConv(hidden, hidden, metadata=metadata, heads=heads)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, batch):
        x_dict = {
            nt: F.relu(self.input_projection[nt](x))
            for nt, x in batch.x_dict.items()
        }
        out = self.conv1(x_dict, batch.edge_index_dict)
        x_dict = {
            nt: F.dropout(
                F.relu(out[nt]) if out.get(nt) is not None else x,
                p=self.dropout,
                training=self.training,
            )
            for nt, x in x_dict.items()
        }
        out = self.conv2(x_dict, batch.edge_index_dict)
        provider = out.get("provider")
        if provider is None:
            provider = x_dict["provider"]
        return self.classifier(provider[: batch["provider"].batch_size])


class HeteroRGCN(nn.Module):
    """RGCNConv on a per-batch homogeneous node matrix with relation IDs."""

    def __init__(self, metadata, in_dims, hidden=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dropout = dropout
        self.input_projection = nn.ModuleDict({
            nt: nn.Linear(in_dims[nt], hidden) for nt in self.node_types
        })
        self.type_embedding = nn.Embedding(len(self.node_types), hidden)
        self.conv1 = RGCNConv(hidden, hidden, num_relations=len(self.edge_types))
        self.conv2 = RGCNConv(hidden, hidden, num_relations=len(self.edge_types))
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, batch):
        x_dict = batch.x_dict
        parts, offsets, offset = [], {}, 0
        for i, nt in enumerate(self.node_types):
            if nt not in x_dict:
                continue
            h = F.relu(self.input_projection[nt](x_dict[nt]))
            h = h + self.type_embedding.weight[i]
            offsets[nt] = offset
            offset += h.size(0)
            parts.append(h)
        h = torch.cat(parts, dim=0)

        edge_indices, relation_ids = [], []
        for relation_id, edge_type in enumerate(self.edge_types):
            edge_index = batch.edge_index_dict.get(edge_type)
            src_type, _, dst_type = edge_type
            if (
                edge_index is None
                or src_type not in offsets
                or dst_type not in offsets
            ):
                continue
            edge_indices.append(torch.stack([
                edge_index[0] + offsets[src_type],
                edge_index[1] + offsets[dst_type],
            ]))
            relation_ids.append(torch.full(
                (edge_index.size(1),), relation_id,
                dtype=torch.long, device=h.device,
            ))

        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_type_ids = torch.cat(relation_ids)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=h.device)
            edge_type_ids = torch.empty((0,), dtype=torch.long, device=h.device)

        h = F.relu(self.conv1(h, edge_index, edge_type_ids))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_type_ids)

        start = offsets["provider"]
        seeds = h[start: start + batch["provider"].batch_size]
        return self.classifier(seeds)
