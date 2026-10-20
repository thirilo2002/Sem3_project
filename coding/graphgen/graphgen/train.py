"""
Training entry point: fits one MixedTypeDiffusion per node type and one
FK sampler per relation, from a REAL `torch_geometric.data.HeteroData`
object that matches schema.py's node/relation names.

Usage:
    from graphgen.schema import build_default_schema
    from graphgen.train import fit_all

    schema = build_default_schema()
    artifacts = fit_all(real_data, schema, device="cuda")
    # artifacts.diffusion_models, artifacts.standardizers, artifacts.fk_samplers, artifacts.vocab_samplers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from .schema import GraphSchema, NodeTypeSpec
from .diffusion import MixedTypeDiffusion, Standardizer
from .fk_sampler import ManyToOneFKSampler, VocabAttachmentSampler


def split_columns(x: torch.Tensor, spec: NodeTypeSpec) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits a (N, feat_dim) real feature tensor into (x_cont, x_cat) per the
    node type's ColumnSpec. x_cat columns are read as integer class ids --
    if your real tensor stores them as floats (e.g. 0.0/1.0 flags), they're
    rounded and cast to long here.
    """
    cont_idx = spec.columns.continuous_idx
    cat_idx = sorted(spec.columns.categorical_idx.keys())
    x_cont = x[:, cont_idx] if cont_idx else torch.zeros(x.shape[0], 0)
    x_cat = x[:, cat_idx].round().long() if cat_idx else torch.zeros(x.shape[0], 0, dtype=torch.long)
    _validate_categorical_range(spec, x_cat)
    return x_cont, x_cat


def _validate_categorical_range(spec: NodeTypeSpec, x_cat: torch.Tensor) -> None:
    """Fails fast on CPU with a readable message if the schema's declared
    num_classes doesn't cover the real data's observed values -- this is
    exactly the mismatch that otherwise surfaces as an opaque
    `nll_loss ... Assertion t >= 0 && t < n_classes failed` CUDA device
    assert, hundreds of training steps later, on the GPU.
    """
    cat_cols = sorted(spec.columns.categorical_idx.keys())
    for pos, col_idx in enumerate(cat_cols):
        k = spec.columns.categorical_idx[col_idx]
        if x_cat.shape[0] == 0:
            continue
        col = x_cat[:, pos]
        lo, hi = int(col.min().item()), int(col.max().item())
        if lo < 0 or hi >= k:
            raise ValueError(
                f"[{spec.name}] column index {col_idx}: schema declares "
                f"{k} classes, but the real data has values in [{lo}, {hi}]. "
                f"Fix schema.py (set categorical_idx[{col_idx}] = {hi + 1}), "
                f"or better, build your schema with "
                f"`schema.build_schema_from_data(real_data, categorical_columns=...)` "
                f"which infers this automatically instead of guessing."
            )


def train_node_diffusion(
    spec: NodeTypeSpec,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 2e-4,
    T: int = 1000,
    cond_dim: int = 0,
    cond: torch.Tensor = None,  # (N, cond_dim), aligned row-for-row with x_cont/x_cat
    device: str = "cpu",
) -> Tuple[MixedTypeDiffusion, Standardizer]:
    std = Standardizer.fit(x_cont)
    x_cont_n = std.transform(x_cont)

    model = MixedTypeDiffusion(spec, T=T, cond_dim=cond_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if cond_dim > 0:
        assert cond is not None and cond.shape[0] == x_cont_n.shape[0], \
            "cond must be provided and row-aligned when cond_dim > 0"
        ds = TensorDataset(x_cont_n, x_cat, cond)
    else:
        ds = TensorDataset(x_cont_n, x_cat)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for batch in dl:
            if cond_dim > 0:
                xb_cont, xb_cat, xb_cond = batch
                xb_cond = xb_cond.to(device)
            else:
                xb_cont, xb_cat = batch
                xb_cond = None
            xb_cont, xb_cat = xb_cont.to(device), xb_cat.to(device)
            opt.zero_grad()
            loss = model.loss(xb_cont, xb_cat, cond=xb_cond)
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"[{spec.name}] epoch {ep+1}/{epochs}  loss={total / max(1, len(dl)):.4f}")
    model.eval()
    return model, std


@dataclass
class FittedArtifacts:
    diffusion_models: Dict[str, MixedTypeDiffusion] = field(default_factory=dict)
    standardizers: Dict[str, Standardizer] = field(default_factory=dict)
    fk_samplers: Dict[Tuple[str, str, str], ManyToOneFKSampler] = field(default_factory=dict)
    vocab_samplers: Dict[Tuple[str, str, str], VocabAttachmentSampler] = field(default_factory=dict)


def fit_all(
    real_data,  # torch_geometric.data.HeteroData
    schema: GraphSchema,
    independent_types: List[str] = ("provider", "physician", "diagnosis", "procedure", "beneficiary"),
    epochs: int = 50,
    device: str = "cpu",
    n_clusters: int = 16,
) -> FittedArtifacts:
    art = FittedArtifacts()

    # 1. unconditional diffusion models for the independent "dimension" tables
    for ntype in independent_types:
        spec = schema.node_types[ntype]
        x_cont, x_cat = split_columns(real_data[ntype].x, spec)
        model, std = train_node_diffusion(spec, x_cont, x_cat, epochs=epochs, device=device)
        art.diffusion_models[ntype] = model
        art.standardizers[ntype] = std

    # 2. claim: conditioned on its cluster id (from beneficiary FK sampler),
    #    one-hot encoded as the conditioning vector
    for rel in schema.forward_relations:
        if rel.dst == "claim" and not rel.many_to_many:
            # e.g. beneficiary -has_claim-> claim : fit FK sampler using
            # beneficiary attrs + the real beneficiary->claim edge_index
            parent_spec = schema.node_types[rel.src]
            parent_x_cont, _ = split_columns(real_data[rel.src].x, parent_spec)
            parent_std = art.standardizers[rel.src]
            parent_feats = parent_std.transform(parent_x_cont)

            edge_index = real_data[rel.src, rel.name, rel.dst].edge_index  # (2, E): [parent_idx, child_idx]
            # child_to_parent[child_local_idx] = parent_idx -- build via scatter
            n_children = real_data[rel.dst].num_nodes
            child_to_parent = torch.zeros(n_children, dtype=torch.long)
            child_to_parent[edge_index[1]] = edge_index[0]

            sampler = ManyToOneFKSampler(n_clusters=n_clusters).fit(parent_feats, child_to_parent)
            art.fk_samplers[(rel.src, rel.name, rel.dst)] = sampler

    claim_spec = schema.node_types["claim"]
    x_cont, x_cat = split_columns(real_data["claim"].x, claim_spec)
    # condition claim generation on its beneficiary-cluster (one relation
    # chosen as the primary conditioning signal; extend to concat multiple
    # parent clusters if you want tighter multi-parent correlation)
    bene_rel = schema.relation("beneficiary", "has_claim", "claim")
    bene_sampler = art.fk_samplers[("beneficiary", "has_claim", "claim")]
    edge_index = real_data["beneficiary", "has_claim", "claim"].edge_index
    child_to_parent = torch.zeros(real_data["claim"].num_nodes, dtype=torch.long)
    child_to_parent[edge_index[1]] = edge_index[0]
    claim_cluster_ids = bene_sampler.kmeans.labels_[child_to_parent.numpy()]
    claim_cluster_ids = torch.tensor(claim_cluster_ids, dtype=torch.long)
    cond_onehot = torch.nn.functional.one_hot(claim_cluster_ids, num_classes=n_clusters).float()

    claim_model, claim_std = train_node_diffusion(
        claim_spec, x_cont, x_cat, epochs=epochs, device=device,
        cond_dim=n_clusters, cond=cond_onehot,
    )
    art.diffusion_models["claim"] = claim_model
    art.standardizers["claim"] = claim_std

    # 2b. physician -affiliated_with-> provider : many-to-one FK, physician
    #     is the child, provider the parent. Reuses the already-trained
    #     physician diffusion model's standardizer for cluster features.
    aff_edge_index = real_data["physician", "affiliated_with", "provider"].edge_index  # [physician_idx, provider_idx]
    n_physicians = real_data["physician"].num_nodes
    phys_to_provider = torch.zeros(n_physicians, dtype=torch.long)
    phys_to_provider[aff_edge_index[0]] = aff_edge_index[1]
    prov_x_cont, _ = split_columns(real_data["provider"].x, schema.node_types["provider"])
    prov_feats = art.standardizers["provider"].transform(prov_x_cont)
    # note: ManyToOneFKSampler.fit expects (parent_feats, child_to_parent);
    # here "parent" = provider, "child" = physician
    aff_sampler = ManyToOneFKSampler(n_clusters=n_clusters).fit(prov_feats, phys_to_provider)
    art.fk_samplers[("physician", "affiliated_with", "provider")] = aff_sampler

    # 2c. beneficiary -sees_physician-> physician : many-to-many, reuse the
    #     SAME beneficiary clusters used for has_claim so a beneficiary's
    #     claim volume and physician-visiting pattern stay correlated.
    sees_edge_index = real_data["beneficiary", "sees_physician", "physician"].edge_index
    n_bene = real_data["beneficiary"].num_nodes
    per_bene_phys: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(n_bene)]
    order = torch.argsort(sees_edge_index[0])
    src_sorted, dst_sorted = sees_edge_index[0][order], sees_edge_index[1][order]
    boundaries = torch.nonzero(torch.diff(src_sorted), as_tuple=True)[0] + 1
    starts = torch.cat([torch.tensor([0]), boundaries])
    ends = torch.cat([boundaries, torch.tensor([src_sorted.shape[0]])])
    for s, e in zip(starts.tolist(), ends.tolist()):
        if s == e:
            continue
        per_bene_phys[src_sorted[s].item()] = dst_sorted[s:e]
    bene_cluster_ids = torch.tensor(bene_sampler.kmeans.labels_, dtype=torch.long)
    sees_sampler = VocabAttachmentSampler(vocab_size=real_data["physician"].num_nodes, n_clusters=n_clusters)
    sees_sampler.fit(bene_cluster_ids, per_bene_phys)
    art.vocab_samplers[("beneficiary", "sees_physician", "physician")] = sees_sampler

    # 3. vocabulary attachment samplers: has_dx, has_px, AND serviced_by /
    #    attended_by (claim->provider, claim->physician), which are
    #    structurally many-to-one but are fit with the same frequency-based
    #    VocabAttachmentSampler machinery with cardinality naturally ~1.
    for rel_name, dst in [("serviced_by", "provider"), ("attended_by", "physician")]:
        edge_index = real_data["claim", rel_name, dst].edge_index  # [claim_idx, dst_idx], ~1 per claim
        n_claims = real_data["claim"].num_nodes
        per_claim_ids: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(n_claims)]
        order = torch.argsort(edge_index[0])
        src_sorted, dst_sorted = edge_index[0][order], edge_index[1][order]
        boundaries = torch.nonzero(torch.diff(src_sorted), as_tuple=True)[0] + 1
        starts = torch.cat([torch.tensor([0]), boundaries])
        ends = torch.cat([boundaries, torch.tensor([src_sorted.shape[0]])])
        for s, e in zip(starts.tolist(), ends.tolist()):
            if s == e:
                continue
            per_claim_ids[src_sorted[s].item()] = dst_sorted[s:e]
        vocab_size = real_data[dst].num_nodes
        sampler = VocabAttachmentSampler(vocab_size=vocab_size, n_clusters=n_clusters)
        sampler.fit(claim_cluster_ids, per_claim_ids)
        art.vocab_samplers[("claim", rel_name, dst)] = sampler

    for rel in schema.forward_relations:
        if rel.many_to_many and rel.dst in ("diagnosis", "procedure"):
            edge_index = real_data["claim", rel.name, rel.dst].edge_index
            n_claims = real_data["claim"].num_nodes
            per_claim_ids: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(n_claims)]
            # group edges by claim
            order = torch.argsort(edge_index[0])
            src_sorted, dst_sorted = edge_index[0][order], edge_index[1][order]
            boundaries = torch.nonzero(torch.diff(src_sorted), as_tuple=True)[0] + 1
            starts = torch.cat([torch.tensor([0]), boundaries])
            ends = torch.cat([boundaries, torch.tensor([src_sorted.shape[0]])])
            for s, e in zip(starts.tolist(), ends.tolist()):
                if s == e:
                    continue
                claim_idx = src_sorted[s].item()
                per_claim_ids[claim_idx] = dst_sorted[s:e]

            vocab_size = real_data[rel.dst].num_nodes
            sampler = VocabAttachmentSampler(vocab_size=vocab_size, n_clusters=n_clusters)
            sampler.fit(claim_cluster_ids, per_claim_ids)
            art.vocab_samplers[("claim", rel.name, rel.dst)] = sampler

    return art
