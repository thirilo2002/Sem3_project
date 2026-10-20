"""
Full generation pipeline: fitted artifacts (from train.fit_all) -> a brand
new synthetic `torch_geometric.data.HeteroData` object matching your schema.

No adjacency matrix is ever built. Edge sets are produced as flat index
arrays straight from the FK/vocab samplers (repeat_interleave + per-row
draws), which is what makes this tractable at claim=558K scale.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch_geometric.data import HeteroData

from .schema import GraphSchema, NODE_COUNTS
from .train import FittedArtifacts


def _reassemble_features(spec, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
    """Interleaves generated continuous + categorical columns back into the
    original column order defined by spec.columns."""
    n = x_cont.shape[0] if x_cont.shape[-1] > 0 else x_cat.shape[0]
    out = torch.zeros(n, spec.feat_dim)
    cont_idx = spec.columns.continuous_idx
    cat_idx = sorted(spec.columns.categorical_idx.keys())
    if cont_idx:
        out[:, cont_idx] = x_cont
    if cat_idx:
        out[:, cat_idx] = x_cat.float()
    return out


def generate_node_table(
    ntype: str, spec, model, std, n: int, device: str = "cpu", cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_cont, x_cat = model.sample(n, cond=cond, device=device)
    x_cont = std.inverse(x_cont.cpu())
    return _reassemble_features(spec, x_cont, x_cat.cpu())


def generate_graph(
    art: FittedArtifacts,
    schema: GraphSchema,
    n_nodes: Optional[Dict[str, int]] = None,
    n_clusters: int = 16,
    device: str = "cpu",
) -> HeteroData:
    n_nodes = n_nodes or dict(NODE_COUNTS)
    data = HeteroData()

    # --- 1. independent dimension tables ---
    independent_types = ["provider", "physician", "diagnosis", "procedure", "beneficiary"]
    for ntype in independent_types:
        spec = schema.node_types[ntype]
        model, std = art.diffusion_models[ntype], art.standardizers[ntype]
        data[ntype].x = generate_node_table(ntype, spec, model, std, n_nodes[ntype], device=device)
        data[ntype].num_nodes = n_nodes[ntype]

    # --- 2. beneficiary -> claim : sample claim count per beneficiary,
    #        then generate claim attrs conditioned on beneficiary cluster ---
    bene_sampler = art.fk_samplers[("beneficiary", "has_claim", "claim")]
    bene_feats_std = art.standardizers["beneficiary"].transform(
        data["beneficiary"].x[:, schema.node_types["beneficiary"].columns.continuous_idx]
    )
    bene_cluster_ids, bene_degrees = bene_sampler.assign(bene_feats_std)
    claim_to_bene, claim_cluster_ids = bene_sampler.expand_to_child_index(bene_cluster_ids, bene_degrees)
    n_claims = claim_to_bene.shape[0]

    cond_onehot = torch.nn.functional.one_hot(claim_cluster_ids, num_classes=n_clusters).float()
    claim_spec = schema.node_types["claim"]
    claim_model, claim_std = art.diffusion_models["claim"], art.standardizers["claim"]
    data["claim"].x = generate_node_table(
        "claim", claim_spec, claim_model, claim_std, n_claims, device=device, cond=cond_onehot.to(device),
    )
    data["claim"].num_nodes = n_claims

    data["beneficiary", "has_claim", "claim"].edge_index = torch.stack(
        [claim_to_bene, torch.arange(n_claims)]
    )
    data["claim", "rev_has_claim", "beneficiary"].edge_index = \
        data["beneficiary", "has_claim", "claim"].edge_index.flip(0)

    # --- 3. claim -> provider, claim -> physician : many-to-one FK,
    #        sampled directly from each relation's degree/cluster model
    #        (here: one provider & one physician per claim, drawn from the
    #        provider/physician population weighted by the fitted sampler) ---
    # Simplification note: claim->provider and claim->physician are at most
    # one-parent-per-claim in most claims datasets. We reuse the claim's own
    # cluster id (from the beneficiary FK) to pick a provider/physician whose
    # OWN cluster (fit independently below) is compatible, via the same
    # ManyToOneFKSampler machinery but inverted: provider/physician are the
    # "parent" and claim is the "child" with degree 1 by construction, so we
    # instead sample provider_id / physician_id per claim directly from the
    # per-cluster empirical id-frequency table captured by VocabAttachmentSampler
    # with cardinality fixed to 1. See train.py if you want full two-sided
    # cluster matching instead of this frequency-based assignment.
    for rel_name, dst in [("serviced_by", "provider"), ("attended_by", "physician")]:
        vocab_sampler = art.vocab_samplers.get(("claim", rel_name, dst))
        if vocab_sampler is None:
            raise KeyError(
                f"No vocab sampler fitted for claim->{dst} ({rel_name}). "
                f"Fit one in train.fit_all with cardinality fixed to 1 per claim."
            )
        chosen = vocab_sampler.sample(claim_cluster_ids)  # list of length-1 arrays (mostly)
        dst_idx = torch.tensor([c[0] if len(c) > 0 else 0 for c in chosen], dtype=torch.long)
        data["claim", rel_name, dst].edge_index = torch.stack([torch.arange(n_claims), dst_idx])
        data[dst, f"rev_{rel_name}", "claim"].edge_index = data["claim", rel_name, dst].edge_index.flip(0)

    # --- 4. claim <-> diagnosis, claim <-> procedure : many-to-many vocab attachment ---
    for rel_name, dst in [("has_dx", "diagnosis"), ("has_px", "procedure")]:
        vocab_sampler = art.vocab_samplers[("claim", rel_name, dst)]
        chosen = vocab_sampler.sample(claim_cluster_ids)
        src_list, dst_list = [], []
        for claim_idx, ids in enumerate(chosen):
            if len(ids) == 0:
                continue
            src_list.append(torch.full((len(ids),), claim_idx, dtype=torch.long))
            dst_list.append(torch.as_tensor(ids, dtype=torch.long))
        src = torch.cat(src_list) if src_list else torch.empty(0, dtype=torch.long)
        dst_ = torch.cat(dst_list) if dst_list else torch.empty(0, dtype=torch.long)
        data["claim", rel_name, dst].edge_index = torch.stack([src, dst_])
        data[dst, f"rev_{rel_name}", "claim"].edge_index = data["claim", rel_name, dst].edge_index.flip(0)

    # --- 5. physician -> provider affiliation ---
    aff_sampler = art.fk_samplers[("physician", "affiliated_with", "provider")]
    phys_feats_std = art.standardizers["physician"].transform(
        data["physician"].x[:, schema.node_types["physician"].columns.continuous_idx]
    )
    phys_cluster_ids, phys_degrees = aff_sampler.assign(phys_feats_std)
    # here the sampler was fit as (parent=provider, child=physician), degree=
    # "how many physicians this provider has" -- so invert: assign each
    # physician to exactly one provider by sampling from the provider
    # population proportional to the fitted degree distribution per cluster.
    # Simplest consistent approach: round-trip through expand_to_child_index
    # using providers as the "parent" axis (n=n_providers) instead.
    n_providers = data["provider"].num_nodes
    prov_feats_std = art.standardizers["provider"].transform(
        data["provider"].x[:, schema.node_types["provider"].columns.continuous_idx]
    )
    prov_cluster_ids, prov_degrees = aff_sampler.assign(prov_feats_std)
    phys_to_prov, _ = aff_sampler.expand_to_child_index(prov_cluster_ids, prov_degrees)
    # expand_to_child_index may produce more/fewer than n_physicians rows;
    # trim or pad by resampling providers uniformly to match n_physicians exactly.
    n_physicians = data["physician"].num_nodes
    if phys_to_prov.shape[0] > n_physicians:
        phys_to_prov = phys_to_prov[:n_physicians]
    elif phys_to_prov.shape[0] < n_physicians:
        pad = torch.randint(0, n_providers, (n_physicians - phys_to_prov.shape[0],))
        phys_to_prov = torch.cat([phys_to_prov, pad])
    data["physician", "affiliated_with", "provider"].edge_index = torch.stack(
        [torch.arange(n_physicians), phys_to_prov]
    )
    data["provider", "rev_affiliated_with", "physician"].edge_index = \
        data["physician", "affiliated_with", "provider"].edge_index.flip(0)

    # --- 6. beneficiary -> physician (sees_physician), many-to-many ---
    sees_sampler = art.vocab_samplers[("beneficiary", "sees_physician", "physician")]
    chosen = sees_sampler.sample(bene_cluster_ids)
    src_list, dst_list = [], []
    for bene_idx, ids in enumerate(chosen):
        if len(ids) == 0:
            continue
        src_list.append(torch.full((len(ids),), bene_idx, dtype=torch.long))
        dst_list.append(torch.as_tensor(ids, dtype=torch.long))
    src = torch.cat(src_list) if src_list else torch.empty(0, dtype=torch.long)
    dst_ = torch.cat(dst_list) if dst_list else torch.empty(0, dtype=torch.long)
    data["beneficiary", "sees_physician", "physician"].edge_index = torch.stack([src, dst_])
    data["physician", "rev_sees_physician", "beneficiary"].edge_index = \
        data["beneficiary", "sees_physician", "physician"].edge_index.flip(0)

    return data
