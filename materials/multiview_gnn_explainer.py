"""
Custom GNNExplainer for a Multi-View Heterogeneous GCN
========================================================

Explains WHY a given node (e.g. a patient/provider) was flagged as fraud, by
learning/searching which edges, in which of the three views (topology,
feature, semantic), mattered most for the prediction.

Two variants are provided:

  A. GradientMaskExplainer   -> works when the full path (graph -> Z -> score)
                                 is differentiable, e.g. explaining the softmax
                                 classification head trained jointly with the GCN.

  B. PerturbationExplainer   -> works with ANY downstream classifier, including
                                 non-differentiable ones like Random Forest,
                                 XGBoost, or SVM. This is the one you want for
                                 your real production pipeline (GCN -> frozen Z
                                 -> RF/XGBoost/SVM).

Both return, per node:
  - an importance score per EDGE, split by view (topology/feature/semantic)
  - an importance score per VIEW overall (which of the 3 graphs mattered most)
  - a human-readable explanation string

Requires: torch, torch_geometric (install these in your own environment;
they are not available in this sandbox, so this file is written carefully
but not executed here -- test it locally before your review).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared data structure for explanation results
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    node_id: int
    node_type: str
    predicted_class: int
    predicted_prob: float
    # edge_index tensor [2, num_kept_edges] and importance score per view
    edge_importance: Dict[str, torch.Tensor] = field(default_factory=dict)
    edge_index_used: Dict[str, torch.Tensor] = field(default_factory=dict)
    view_importance: Dict[str, float] = field(default_factory=dict)
    top_neighbors: List[Tuple[str, int, float]] = field(default_factory=list)  # (view, neighbor_id, score)

    def summary(self, k: int = 5) -> str:
        """Human-readable explanation, ready to show in your XAI slide/demo."""
        views_sorted = sorted(self.view_importance.items(), key=lambda x: -x[1])
        top_view, top_view_score = views_sorted[0]

        lines = [
            f"Node {self.node_id} ({self.node_type}) -> "
            f"predicted class {self.predicted_class} "
            f"(risk score {self.predicted_prob:.3f})",
            "",
            "View contribution (higher = more influence on this prediction):",
        ]
        for view, score in views_sorted:
            lines.append(f"  - {view:<10s}: {score:.3f}")

        lines.append("")
        lines.append(f"Primary driver: the {top_view} view "
                      f"(contribution {top_view_score:.3f}).")

        top_k_neighbors = sorted(self.top_neighbors, key=lambda x: -x[2])[:k]
        if top_k_neighbors:
            lines.append("")
            lines.append("Most influential connections:")
            for view, nbr, score in top_k_neighbors:
                lines.append(f"  - via {view} view -> neighbor node {nbr} "
                              f"(importance {score:.3f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# A. Gradient-based mask explainer
#    (use when explaining a differentiable classification head)
# ---------------------------------------------------------------------------

class GradientMaskExplainer:
    """
    Learns a soft, per-edge mask (one scalar per edge, per view) that, when
    used to re-weight messages during GCN aggregation, reproduces the
    original prediction as closely as possible while using as FEW edges as
    possible. This is the classic GNNExplainer idea (Ying et al., 2019),
    adapted to operate over three separate graphs at once.

    Requires your model's forward() to accept an `edge_weight` per view,
    e.g.  model(x_dict, edge_index_dict, edge_weight_dict) -> logits
    Most PyG conv layers (GCNConv, SAGEConv, etc.) natively support an
    `edge_weight` argument, so wiring this in is usually a small change to
    your existing multi-view GCN forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 200,
        lr: float = 0.05,
        edge_size_coef: float = 0.005,   # penalizes using MANY edges
        edge_entropy_coef: float = 1.0,  # pushes mask values toward 0 or 1
        init_bias: float = 5.0,          # start masks close to "all edges on"
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.edge_size_coef = edge_size_coef
        self.edge_entropy_coef = edge_entropy_coef
        self.init_bias = init_bias

        # freeze the trained model -- we only optimize the masks
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def _init_masks(self, edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, nn.Parameter]:
        masks = {}
        for view, edge_index in edge_index_dict.items():
            num_edges = edge_index.size(1)
            # small positive init -> sigmoid(init) close to 1 (all edges "on" at start)
            init = torch.randn(num_edges) * 0.1 + self.init_bias
            masks[view] = nn.Parameter(init)
        return masks

    def explain_node(
        self,
        node_id: int,
        node_type: str,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> Explanation:
        """
        edge_index_dict: {'topology': [2, E_t], 'feature': [2, E_f], 'semantic': [2, E_s]}
        All edge_index tensors reference the SAME global node indexing scheme
        used by your model (i.e. consistent with x_dict).
        """
        masks = self._init_masks(edge_index_dict)
        optimizer = torch.optim.Adam(masks.values(), lr=self.lr)

        # get the model's original (unmasked) prediction, to know the target class
        with torch.no_grad():
            orig_logits = self.model(x_dict, edge_index_dict)
            orig_probs = F.softmax(orig_logits[node_id].unsqueeze(0), dim=-1)
            if target_class is None:
                target_class = int(orig_probs.argmax(dim=-1).item())

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            edge_weight_dict = {
                view: torch.sigmoid(mask) for view, mask in masks.items()
            }

            logits = self.model(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
            log_probs = F.log_softmax(logits[node_id].unsqueeze(0), dim=-1)

            # (1) fidelity term: keep the masked prediction close to target class
            pred_loss = -log_probs[0, target_class]

            # (2) sparsity term: penalize using many edges (per view)
            size_loss = sum(torch.sigmoid(m).sum() for m in masks.values())

            # (3) entropy term: push each mask value toward 0 or 1 (discrete-like)
            entropy_loss = 0.0
            for m in masks.values():
                p = torch.sigmoid(m)
                entropy_loss = entropy_loss + (
                    -p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
                ).mean()

            loss = (
                pred_loss
                + self.edge_size_coef * size_loss
                + self.edge_entropy_coef * entropy_loss
            )
            loss.backward()
            optimizer.step()

        return self._build_explanation(
            node_id, node_type, masks, edge_index_dict, target_class, orig_probs
        )

    def _build_explanation(
        self, node_id, node_type, masks, edge_index_dict, target_class, orig_probs
    ) -> Explanation:
        exp = Explanation(
            node_id=node_id,
            node_type=node_type,
            predicted_class=target_class,
            predicted_prob=float(orig_probs[0, target_class]),
        )

        total_importance = 0.0
        view_totals = {}
        for view, mask in masks.items():
            weights = torch.sigmoid(mask).detach()
            exp.edge_importance[view] = weights
            exp.edge_index_used[view] = edge_index_dict[view]
            view_totals[view] = float(weights.sum())
            total_importance += view_totals[view]

            edge_index = edge_index_dict[view]
            # collect edges touching this node, ranked by mask weight
            mask_at_node = (edge_index[1] == node_id) | (edge_index[0] == node_id)
            idx = mask_at_node.nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                nbr = dst if src == node_id else src
                exp.top_neighbors.append((view, nbr, float(weights[i])))

        # normalize view contributions to sum to 1, for readability
        for view in view_totals:
            exp.view_importance[view] = (
                view_totals[view] / total_importance if total_importance > 0 else 0.0
            )

        return exp


# ---------------------------------------------------------------------------
# B. Perturbation-based explainer
#    (works with ANY downstream classifier: RF, XGBoost, SVM, ...)
# ---------------------------------------------------------------------------

class PerturbationExplainer:
    """
    Model-agnostic explainer for the two-stage pipeline:
        frozen GCN  ->  embedding Z  ->  external classifier.predict_proba()

    Idea: for the target node, repeatedly DROP subsets of edges (per view),
    recompute Z through the frozen GCN, re-score with the real classifier,
    and measure how much the fraud probability drops. Edges whose removal
    causes the biggest drop are the most important edges. No gradients,
    no differentiability requirement -- works with sklearn/XGBoost directly.

    This is slower (many forward passes) but matches your ACTUAL deployed
    pipeline exactly, rather than a differentiable proxy of it.
    """

    def __init__(
        self,
        gcn_forward_fn: Callable[
            [Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor
        ],
        classifier_predict_fn: Callable[[torch.Tensor], float],
        node_id: int,
        node_type: str,
    ):
        """
        gcn_forward_fn(x_dict, edge_index_dict) -> Z  [num_nodes, embed_dim]
            Your trained, FROZEN multi-view GCN's forward pass up to the
            fused embedding Z (i.e. everything through attention fusion,
            stop before the classifier).

        classifier_predict_fn(z_row) -> float
            Wraps your real classifier, e.g.:
                lambda z: rf_model.predict_proba(z.numpy().reshape(1, -1))[0, 1]
        """
        self.gcn_forward_fn = gcn_forward_fn
        self.classifier_predict_fn = classifier_predict_fn
        self.node_id = node_id
        self.node_type = node_type

    def _score(self, x_dict, edge_index_dict) -> float:
        with torch.no_grad():
            Z = self.gcn_forward_fn(x_dict, edge_index_dict)
            return self.classifier_predict_fn(Z[self.node_id])

    def _edges_touching_node(self, edge_index: torch.Tensor) -> torch.Tensor:
        mask = (edge_index[0] == self.node_id) | (edge_index[1] == self.node_id)
        return mask.nonzero(as_tuple=True)[0]

    def explain_node(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[str, torch.Tensor],
        top_k: int = 10,
    ) -> Explanation:
        base_score = self._score(x_dict, edge_index_dict)
        pred_class = int(base_score >= 0.5)

        exp = Explanation(
            node_id=self.node_id,
            node_type=self.node_type,
            predicted_class=pred_class,
            predicted_prob=base_score,
        )

        # --- View-level importance: drop one entire view's edges touching
        #     this node, measure the score drop ---
        view_drop = {}
        for view, edge_index in edge_index_dict.items():
            touching = self._edges_touching_node(edge_index)
            if len(touching) == 0:
                view_drop[view] = 0.0
                continue
            keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
            keep_mask[touching] = False
            perturbed = dict(edge_index_dict)
            perturbed[view] = edge_index[:, keep_mask]

            score_without_view = self._score(x_dict, perturbed)
            view_drop[view] = max(0.0, base_score - score_without_view)

        total_drop = sum(view_drop.values())
        for view in view_drop:
            exp.view_importance[view] = (
                view_drop[view] / total_drop if total_drop > 0 else 0.0
            )

        # --- Edge-level importance WITHIN the most influential view(s):
        #     drop one edge at a time, measure the score drop ---
        for view, edge_index in edge_index_dict.items():
            touching = self._edges_touching_node(edge_index)
            edge_scores = torch.zeros(len(touching))
            for rank, i in enumerate(touching.tolist()):
                keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
                keep_mask[i] = False
                perturbed = dict(edge_index_dict)
                perturbed[view] = edge_index[:, keep_mask]

                score_without_edge = self._score(x_dict, perturbed)
                edge_scores[rank] = max(0.0, base_score - score_without_edge)

                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                nbr = dst if src == self.node_id else src
                exp.top_neighbors.append((view, nbr, float(edge_scores[rank])))

            exp.edge_importance[view] = edge_scores
            exp.edge_index_used[view] = edge_index[:, touching]

        exp.top_neighbors = sorted(exp.top_neighbors, key=lambda t: -t[2])[:top_k]
        return exp


# ---------------------------------------------------------------------------
# Fidelity metric -- use this to VALIDATE your explainer (whichever variant)
# rather than just trusting it looks plausible. This directly answers the
# "how do you know your explanations are faithful" question from your review.
# ---------------------------------------------------------------------------

def fidelity_minus(
    base_score: float,
    score_fn: Callable[[Dict[str, torch.Tensor]], float],
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[str, torch.Tensor],
    important_edges: Dict[str, torch.Tensor],  # edge indices (positions) per view deemed important
) -> float:
    """
    Fidelity- : remove the edges the explainer says are IMPORTANT.
    A GOOD explainer should cause a LARGE score drop here
    (i.e. fidelity_minus should be HIGH -- removing important stuff hurts a lot).
    """
    perturbed = {}
    for view, edge_index in edge_index_dict.items():
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        if view in important_edges:
            keep_mask[important_edges[view]] = False
        perturbed[view] = edge_index[:, keep_mask]

    perturbed_score = score_fn(perturbed)
    return base_score - perturbed_score


def fidelity_plus(
    base_score: float,
    score_fn: Callable[[Dict[str, torch.Tensor]], float],
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[str, torch.Tensor],
    important_edges: Dict[str, torch.Tensor],
) -> float:
    """
    Fidelity+ : keep ONLY the edges the explainer says are important, drop
    everything else. A GOOD explainer should cause a SMALL score drop here
    (i.e. fidelity_plus should be LOW -- the "important" edges alone are
    nearly enough to reproduce the original prediction).
    """
    perturbed = {}
    for view, edge_index in edge_index_dict.items():
        if view in important_edges and len(important_edges[view]) > 0:
            perturbed[view] = edge_index[:, important_edges[view]]
        else:
            perturbed[view] = edge_index[:, :0]  # keep nothing from this view

    perturbed_score = score_fn(perturbed)
    return base_score - perturbed_score


# ---------------------------------------------------------------------------
# Example usage (adapt to your actual model / classifier objects)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    This block will not run as-is -- it's a template showing how to wire
    your trained model into the two explainer classes above.
    """

    # --- Variant A: explaining the differentiable softmax head ---
    # explainer_a = GradientMaskExplainer(model=my_trained_mhgsl_model, epochs=200)
    # result_a = explainer_a.explain_node(
    #     node_id=42, node_type="patient",
    #     x_dict=x_dict, edge_index_dict=edge_index_dict,
    # )
    # print(result_a.summary())

    # --- Variant B: explaining the real RF/XGBoost/SVM pipeline ---
    # def gcn_forward(x_dict, edge_index_dict):
    #     with torch.no_grad():
    #         return my_trained_mhgsl_model.embed(x_dict, edge_index_dict)  # returns Z
    #
    # def classifier_predict(z_row):
    #     return float(rf_model.predict_proba(z_row.numpy().reshape(1, -1))[0, 1])
    #
    # explainer_b = PerturbationExplainer(
    #     gcn_forward_fn=gcn_forward,
    #     classifier_predict_fn=classifier_predict,
    #     node_id=42, node_type="patient",
    # )
    # result_b = explainer_b.explain_node(x_dict, edge_index_dict)
    # print(result_b.summary())
    pass
