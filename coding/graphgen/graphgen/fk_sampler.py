"""
Cluster-guided foreign-key (FK) assignment, ClavaDDPM-style
(Pang et al., "ClavaDDPM: Multi-relational Data Synthesis with
Cluster-guided Diffusion Models", NeurIPS 2024, arXiv:2405.17724).

Why this exists: with claim=558K and beneficiary=138K nodes, you cannot
diffuse a dense adjacency matrix (that's O(N*M) just to represent it). The
trick is to never represent "who connects to whom" as a matrix at all --
instead, learn:

  1. a *cluster* label per parent row that summarizes "what kind of children
     does this parent tend to have" (e.g. a beneficiary cluster that implies
     "high claim volume, chronic-condition heavy, sees few physicians"),
  2. a *degree distribution* per cluster (how many claims does a beneficiary
     in this cluster have?),
  3. a child-attribute diffusion model *conditioned on the cluster id*
     (handled by diffusion.MixedTypeDiffusion's `cond` argument).

Generation then becomes: generate parent row -> predict its cluster ->
sample a degree from that cluster's degree distribution -> generate that
many child rows conditioned on the cluster -> write the FK indices directly
as a flat array. No adjacency matrix is ever built.

This module covers two relation shapes:
  - ManyToOneFKSampler: each child row has exactly one parent (claim->
    beneficiary, claim->provider, claim->physician, physician->provider).
  - VocabAttachmentSampler: many-to-many attachment to a fixed, reusable
    vocabulary table (claim<->diagnosis, claim<->procedure, beneficiary<->
    physician), where cardinality varies per child and items are drawn from
    a learned per-cluster frequency distribution over vocabulary ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.cluster import KMeans
except ImportError as e:  # pragma: no cover
    KMeans = None


def _require_sklearn():
    if KMeans is None:
        raise ImportError(
            "scikit-learn is required for cluster-guided FK assignment "
            "(`pip install scikit-learn`)."
        )


class ClusterClassifier(nn.Module):
    """Small MLP: parent attribute vector -> cluster logits. Used at
    generation time to assign a cluster to freshly-sampled (synthetic)
    parent rows, since they have no observed children to cluster on."""

    def __init__(self, in_dim: int, n_clusters: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_clusters),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class ManyToOneFKSampler:
    """Learns, for one (parent_type -> child_type) FK relation, how to
    assign child rows to parent rows while preserving parent/child attribute
    correlation and the real degree distribution.
    """
    n_clusters: int = 16
    kmeans: Optional["KMeans"] = None
    classifier: Optional[ClusterClassifier] = None
    degree_dist_per_cluster: Dict[int, np.ndarray] = None  # cluster -> array of observed degrees

    def fit(
        self,
        parent_feats: torch.Tensor,     # (P, d_parent) real parent attrs (standardized)
        child_to_parent: torch.Tensor,  # (C,) long, index into parent_feats for each real child
        epochs: int = 50,
        lr: float = 1e-3,
    ):
        _require_sklearn()
        P = parent_feats.shape[0]
        degree = torch.bincount(child_to_parent, minlength=P).numpy()  # (P,)

        # cluster parents on [own attrs | own degree] so clusters encode
        # "what kind of children volume/profile this parent has"
        cluster_input = np.concatenate(
            [parent_feats.numpy(), degree[:, None].astype(np.float32)], axis=1
        )
        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
        labels = km.fit_predict(cluster_input)
        self.kmeans = km

        self.degree_dist_per_cluster = {
            c: degree[labels == c] if (labels == c).any() else np.array([0])
            for c in range(self.n_clusters)
        }

        # train classifier: parent attrs (WITHOUT degree, since at generation
        # time synthetic parents have no observed degree yet) -> cluster label
        clf = ClusterClassifier(parent_feats.shape[1], self.n_clusters)
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        y = torch.tensor(labels, dtype=torch.long)
        for _ in range(epochs):
            opt.zero_grad()
            logits = clf(parent_feats)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        self.classifier = clf
        return self

    @torch.no_grad()
    def assign(self, synthetic_parent_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given newly generated parent rows, predict their cluster and
        sample a degree for each. Returns (cluster_ids, degrees) both (P,).
        """
        logits = self.classifier(synthetic_parent_feats)
        cluster_ids = torch.distributions.Categorical(logits=logits).sample()
        degrees = torch.zeros(synthetic_parent_feats.shape[0], dtype=torch.long)
        for i, c in enumerate(cluster_ids.tolist()):
            pool = self.degree_dist_per_cluster[c]
            degrees[i] = int(np.random.choice(pool))
        return cluster_ids, degrees

    def expand_to_child_index(self, cluster_ids: torch.Tensor, degrees: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Turns per-parent (cluster, degree) into flat child-level arrays:
        parent_idx repeated `degree` times, and the matching cluster id per
        child row (to condition the child attribute diffusion model on).
        No adjacency matrix -- just repeat_interleave.
        """
        parent_idx = torch.arange(cluster_ids.shape[0])
        child_parent_idx = parent_idx.repeat_interleave(degrees)
        child_cluster_idx = cluster_ids.repeat_interleave(degrees)
        return child_parent_idx, child_cluster_idx


@dataclass
class VocabAttachmentSampler:
    """Many-to-many attachment of child rows (claims) to a fixed, reusable
    vocabulary (diagnosis / procedure / physician). Learns, per cluster id
    (reuse the child's own cluster id from whichever ManyToOneFKSampler it
    came from -- e.g. claim's beneficiary-cluster), a frequency distribution
    over vocabulary ids, plus a cardinality (how many codes per claim)
    distribution.
    """
    vocab_size: int
    n_clusters: int
    freq_per_cluster: Dict[int, np.ndarray] = None   # cluster -> (vocab_size,) probs
    cardinality_per_cluster: Dict[int, np.ndarray] = None  # cluster -> array of observed counts

    def fit(
        self,
        child_cluster_ids: torch.Tensor,          # (C,) cluster id per real child row (e.g. claim)
        child_vocab_edges: List[torch.Tensor],     # list length C, each a LongTensor of vocab ids attached
    ):
        self.freq_per_cluster = {}
        self.cardinality_per_cluster = {}
        clusters = child_cluster_ids.numpy()
        for c in range(self.n_clusters):
            mask = clusters == c
            idxs = np.where(mask)[0]
            counts = np.zeros(self.vocab_size, dtype=np.float64)
            cards = []
            for i in idxs:
                ids = child_vocab_edges[i]
                cards.append(len(ids))
                if len(ids) > 0:
                    counts[ids.numpy()] += 1
            if counts.sum() == 0:
                counts[:] = 1.0  # fallback: uniform
            self.freq_per_cluster[c] = counts / counts.sum()
            self.cardinality_per_cluster[c] = np.array(cards) if len(cards) > 0 else np.array([1])
        return self

    def sample(self, child_cluster_ids: torch.Tensor) -> List[np.ndarray]:
        """For each (synthetic) child row, sample a cardinality from its
        cluster's distribution, then draw that many vocab ids without
        replacement weighted by the cluster's frequency distribution.
        Returns a list of arrays of vocab ids, one per child row.
        """
        out = []
        for c in child_cluster_ids.tolist():
            card_pool = self.cardinality_per_cluster.get(c, np.array([1]))
            k = int(np.random.choice(card_pool))
            k = max(0, min(k, self.vocab_size))
            if k == 0:
                out.append(np.array([], dtype=np.int64))
                continue
            probs = self.freq_per_cluster.get(c, np.full(self.vocab_size, 1.0 / self.vocab_size))
            chosen = np.random.choice(self.vocab_size, size=k, replace=False, p=probs)
            out.append(chosen)
        return out
