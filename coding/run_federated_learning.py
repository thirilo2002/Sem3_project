
"""Federated comparison of HeteroSAGE, HeteroHGT, and HeteroRGCN."""

import copy
import gc
import os
import random
from collections import defaultdict, deque
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from models.gnn_models.config import (
    SEED,
    OUTPUT_DIR,
    HIDDEN_DIM,
    DROPOUT,
    HGT_HEADS,
    EPOCHS,
    BATCH_SIZE,
    NUM_NEIGHBORS,
    FORCE_CPU,
    COLORS,
    INK_PRIMARY,
    INK_SECONDARY,
    GRIDLINE,
    BASELINE,
    SURFACE,
    GRAPH_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
)
from models.gnn_models.models import HeteroSAGE, HeteroHGT, HeteroRGCN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_CLIENTS = 5
LOCAL_EPOCHS = 40
FEDERATED_ROUNDS = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------
# FedAvg: user's aggregation function
# ---------------------------------------------------------

def fedavg(local_states, client_sizes):
    """Weighted FedAvg for compatible model state dictionaries."""
    total_size = sum(client_sizes)

    if total_size == 0:
        raise ValueError("No client has training examples.")

    weights = [n / total_size for n in client_sizes]
    global_state = copy.deepcopy(local_states[0])

    for key in global_state:
        if torch.is_floating_point(global_state[key]):
            global_state[key] = sum(
                state[key].to(global_state[key].device) * weight
                for state, weight in zip(local_states, weights)
            )
        else:
            global_state[key] = local_states[0][key].clone()

    return global_state

def fedadagrad(global_state, local_states, client_sizes, accumulator, lr=0.01, tau=1e-8):
    """Weighted FedAdagrad server update; accumulator persists across rounds."""
    total_size = sum(client_sizes)
    if total_size == 0:
        raise ValueError("No client has training examples.")

    weights = [n / total_size for n in client_sizes]
    new_state = copy.deepcopy(global_state)

    for key, current in global_state.items():
        if not torch.is_floating_point(current):
            continue

        average_local = sum(
            state[key].to(current.device) * weight
            for state, weight in zip(local_states, weights)
        )
        delta = average_local - current

        if key not in accumulator:
            accumulator[key] = torch.zeros_like(delta)

        accumulator[key].add_(delta.square())
        new_state[key] = current + lr * delta / (
            accumulator[key].sqrt() + tau
        )

    return new_state, accumulator


# ---------------------------------------------------------
# Graph preparation and splitting
# ---------------------------------------------------------

def signed_log1p(x):
    return torch.sign(x) * torch.log1p(x.abs())


def load_graph_and_splits(graph_path=GRAPH_PATH, seed=SEED):
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    for node_type in graph.node_types:
        if "x" not in graph[node_type]:
            raise ValueError(
                f"Missing features for node type '{node_type}'"
            )

    labels = graph["provider"].y.long().view(-1)

    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Expected binary provider labels encoded as 0 and 1.")

    provider_ids = np.arange(graph["provider"].num_nodes)
    label_array = labels.numpy()

    train_ids, temp_ids = train_test_split(
        provider_ids,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=seed,
        stratify=label_array,
    )

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test,
        random_state=seed,
        stratify=label_array[temp_ids],
    )

    train_ids = torch.tensor(train_ids, dtype=torch.long)
    val_ids = torch.tensor(val_ids, dtype=torch.long)
    test_ids = torch.tensor(test_ids, dtype=torch.long)

    # Normalize features. Provider statistics use training providers only.
    for node_type in graph.node_types:
        x = torch.nan_to_num(
            graph[node_type].x.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        x = signed_log1p(x)

        fit_rows = x[train_ids] if node_type == "provider" else x
        mean = fit_rows.mean(dim=0, keepdim=True)
        std = fit_rows.std(dim=0, keepdim=True).clamp(min=1e-6)
        graph[node_type].x = (x - mean) / std

    print(
        f"Providers: {len(provider_ids)} | "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )

    return graph, labels, train_ids, val_ids, test_ids


# ---------------------------------------------------------
# Partition edges across clients.
#
# Reverse relations are paired when exact reverse edge types
# are present. Each original edge is assigned to exactly one
# client. Nodes are induced by the client's assigned edges.
# ---------------------------------------------------------

def make_edge_partitions(graph, num_clients=NUM_CLIENTS, seed=SEED):
    generator = np.random.default_rng(seed)
    edge_types = list(graph.edge_types)
    edge_type_set = set(edge_types)

    # A relation pair must be explicitly declared here if its
    # names/endpoints do not follow the conventional reverse pattern.
    reverse_map = {
        ("beneficiary", "has_claim", "claim"):
            ("claim", "belongs_to", "beneficiary"),
        ("claim", "belongs_to", "beneficiary"):
            ("beneficiary", "has_claim", "claim"),
        ("claim", "submitted_by", "provider"):
            ("provider", "submitted_claim", "claim"),
        ("provider", "submitted_claim", "claim"):
            ("claim", "submitted_by", "provider"),
        ("claim", "attended_by", "physician"):
            ("physician", "attended_claim", "claim"),
        ("physician", "attended_claim", "claim"):
            ("claim", "attended_by", "physician"),
        ("claim", "operated_by", "physician"):
            ("physician", "operated_claim", "claim"),
        ("physician", "operated_claim", "claim"):
            ("claim", "operated_by", "physician"),
        ("claim", "other_physician", "physician"):
            ("physician", "other_claim", "claim"),
        ("physician", "other_claim", "claim"):
            ("claim", "other_physician", "physician"),
        ("claim", "has_diagnosis", "diagnosis"):
            ("diagnosis", "diagnosed_claim", "claim"),
        ("diagnosis", "diagnosed_claim", "claim"):
            ("claim", "has_diagnosis", "diagnosis"),
    }

    # Each group contains edge indices that must stay together.
    groups = []
    visited = set()

    for edge_type in edge_types:
        if edge_type in visited:
            continue

        reverse_type = reverse_map.get(edge_type)

        if reverse_type in edge_type_set and reverse_type != edge_type:
            src, dst = graph[edge_type].edge_index
            rsrc, rdst = graph[reverse_type].edge_index

            # Build a multiset of reverse edge indices. Deques
            # preserve duplicate edges rather than collapsing them.
            reverse_indices = defaultdict(deque)
            for j, pair in enumerate(zip(rsrc.tolist(), rdst.tolist())):
                reverse_indices[pair].append(j)

            paired = []
            unmatched = []

            for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
                q = reverse_indices.get((v, u))
                if q:
                    paired.append((i, q.popleft()))
                else:
                    unmatched.append(i)

            # If any pair cannot be matched, fail rather than silently
            # claiming exact reverse-edge consistency.
            leftovers = sum(len(q) for q in reverse_indices.values())
            if unmatched or leftovers:
                raise ValueError(
                    f"Reverse edges do not match for {edge_type} and "
                    f"{reverse_type}: unmatched forward={len(unmatched)}, "
                    f"unmatched reverse={leftovers}. Check duplicate edges "
                    "and relation direction before partitioning."
                )

            groups.append((edge_type, reverse_type, paired))
            visited.add(edge_type)
            visited.add(reverse_type)
        else:
            num_edges = graph[edge_type].edge_index.size(1)
            groups.append((edge_type, None, [(i, None) for i in range(num_edges)]))
            visited.add(edge_type)

    # Assign each paired edge unit to one client.
    client_edge_ids = [
        {edge_type: [] for edge_type in edge_types}
        for _ in range(num_clients)
    ]

    for edge_type, reverse_type, pairs in groups:
        order = generator.permutation(len(pairs))
        splits = np.array_split(order, num_clients)

        for client_id, split in enumerate(splits):
            for pair_idx in split:
                forward_idx, reverse_idx = pairs[int(pair_idx)]
                client_edge_ids[client_id][edge_type].append(forward_idx)

                if reverse_type is not None:
                    client_edge_ids[client_id][reverse_type].append(reverse_idx)

    # Verify that every edge index is assigned exactly once.
    for edge_type in edge_types:
        assigned = [
            idx
            for client in client_edge_ids
            for idx in client[edge_type]
        ]
        expected = list(range(graph[edge_type].edge_index.size(1)))
        if sorted(assigned) != expected:
            raise AssertionError(
                f"Edge partition is not exhaustive/disjoint for {edge_type}"
            )

    return client_edge_ids


def build_client_graphs(graph, client_edge_ids):
    """Create compact local graphs and preserve original node IDs."""
    clients = []

    for client_id, edge_ids in enumerate(client_edge_ids):
        local_graph = HeteroData()
        node_id_sets = {nt: set() for nt in graph.node_types}

        for node_type in graph.node_types:
            local_graph[node_type].x = graph[node_type].x[:0].clone()
            if "y" in graph[node_type]:
                local_graph[node_type].y = graph[node_type].y[:0].clone()

        # Collect nodes touched by assigned edges.
        for edge_type in graph.edge_types:
            ids = torch.tensor(edge_ids[edge_type], dtype=torch.long)
            edge_index = graph[edge_type].edge_index[:, ids]
            src_type, _, dst_type = edge_type
            node_id_sets[src_type].update(edge_index[0].tolist())
            node_id_sets[dst_type].update(edge_index[1].tolist())

        # Allocate isolated nodes once so every original vertex is
        # represented in the union of client vertex sets.
        # Procedure nodes, for example, may have no listed relations.
        for node_type in graph.node_types:
            if node_type == graph.node_types[0]:
                pass

        clients.append((local_graph, node_id_sets, edge_ids))

    # Assign isolated/uncovered nodes to the smallest client, without
    # adding any arbitrary node to every client.
    all_used = {nt: set() for nt in graph.node_types}
    for _, node_sets, _ in clients:
        for nt in graph.node_types:
            all_used[nt].update(node_sets[nt])

    for node_type in graph.node_types:
        missing = sorted(
            set(range(graph[node_type].num_nodes)) - all_used[node_type]
        )
        for j, node_id in enumerate(missing):
            client_id = j % len(clients)
            clients[client_id][1][node_type].add(node_id)

    # Materialize compact node ID mappings and local edge indices.
    client_graphs = []
    client_global_ids = []

    for local_graph, node_sets, edge_ids in clients:
        global_ids_by_type = {}
        global_to_local = {}

        for node_type in graph.node_types:
            global_ids = torch.tensor(
                sorted(node_sets[node_type]), dtype=torch.long
            )
            global_ids_by_type[node_type] = global_ids
            global_to_local[node_type] = {
                int(gid): i for i, gid in enumerate(global_ids.tolist())
            }

            local_graph[node_type].x = graph[node_type].x[global_ids].clone()
            local_graph[node_type].global_id = global_ids

            if "y" in graph[node_type]:
                local_graph[node_type].y = graph[node_type].y[global_ids].clone()

        for edge_type in graph.edge_types:
            src_type, relation, dst_type = edge_type
            ids = torch.tensor(edge_ids[edge_type], dtype=torch.long)
            original_edges = graph[edge_type].edge_index[:, ids]

            src_map = global_to_local[src_type]
            dst_map = global_to_local[dst_type]

            if original_edges.numel():
                local_src = torch.tensor(
                    [src_map[int(v)] for v in original_edges[0].tolist()],
                    dtype=torch.long,
                )
                local_dst = torch.tensor(
                    [dst_map[int(v)] for v in original_edges[1].tolist()],
                    dtype=torch.long,
                )
                local_edge_index = torch.stack([local_src, local_dst])
            else:
                local_edge_index = torch.empty((2, 0), dtype=torch.long)

            local_graph[edge_type].edge_index = local_edge_index

            if "edge_attr" in graph[edge_type]:
                local_graph[edge_type].edge_attr = (
                    graph[edge_type].edge_attr[ids].clone()
                )

        client_graphs.append(local_graph)
        client_global_ids.append(global_ids_by_type)

    return client_graphs, client_global_ids


# ---------------------------------------------------------
# Client provider train/validation/test ownership
# ---------------------------------------------------------

def allocate_provider_splits(client_global_ids, train_ids, val_ids, test_ids):
    """Assign each supervised provider to exactly one client."""
    train_set = set(train_ids.tolist())
    val_set = set(val_ids.tolist())
    test_set = set(test_ids.tolist())

    train_local = [[] for _ in client_global_ids]
    val_local = [[] for _ in client_global_ids]
    test_local = [[] for _ in client_global_ids]

    # Each global provider is owned by the first client containing it.
    owner = {}
    for client_id, ids_by_type in enumerate(client_global_ids):
        for global_id in ids_by_type["provider"].tolist():
            owner.setdefault(int(global_id), client_id)

    for global_id in train_set:
        client_id = owner[global_id]
        local_map = {
            int(gid): i
            for i, gid in enumerate(
                client_global_ids[client_id]["provider"].tolist()
            )
        }
        train_local[client_id].append(local_map[global_id])

    for global_id in val_set:
        client_id = owner[global_id]
        local_map = {
            int(gid): i
            for i, gid in enumerate(
                client_global_ids[client_id]["provider"].tolist()
            )
        }
        val_local[client_id].append(local_map[global_id])

    for global_id in test_set:
        client_id = owner[global_id]
        local_map = {
            int(gid): i
            for i, gid in enumerate(
                client_global_ids[client_id]["provider"].tolist()
            )
        }
        test_local[client_id].append(local_map[global_id])

    return (
        [torch.tensor(v, dtype=torch.long) for v in train_local],
        [torch.tensor(v, dtype=torch.long) for v in val_local],
        [torch.tensor(v, dtype=torch.long) for v in test_local],
    )


def make_loader(graph, ids, shuffle):
    if len(ids) == 0:
        return None
    return NeighborLoader(
        graph,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=("provider", ids),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def get_class_weights(labels, train_ids, device):
    counts = torch.bincount(labels[train_ids], minlength=2).float()
    return (counts.sum() / (2 * counts.clamp(min=1))).to(device)


# ---------------------------------------------------------
# Local client training and evaluation
# ---------------------------------------------------------

def train_local_model(
    model,
    loader,
    class_weights,
    device,
    epochs=LOCAL_EPOCHS,
):
    if loader is None:
        return model

    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            seeds = batch["provider"].batch_size
            y = batch["provider"].y[:seeds]

            optimizer.zero_grad()
            logits = model(batch)

            if logits.shape != (seeds, 2):
                raise ValueError(
                    f"Expected ({seeds}, 2) logits; got {tuple(logits.shape)}"
                )

            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()

    return model


@torch.no_grad()
def evaluate_model(model, loaders, class_weights, device):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    all_y = []
    all_prob = []

    for loader in loaders:
        if loader is None:
            continue

        for batch in loader:
            batch = batch.to(device)
            seeds = batch["provider"].batch_size
            y = batch["provider"].y[:seeds]
            logits = model(batch)

            loss = F.cross_entropy(
                logits, y, weight=class_weights, reduction="sum"
            )
            total_loss += loss.item()
            total_seen += seeds
            all_y.append(y.cpu())
            all_prob.append(logits.softmax(dim=1)[:, 1].cpu())

    if not all_y:
        raise ValueError("Evaluation produced no provider predictions.")

    y_true = torch.cat(all_y).numpy()
    y_prob = torch.cat(all_prob).numpy()
    y_pred = (y_prob >= 0.5).astype(np.int64)
    two_classes = len(np.unique(y_true)) == 2

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Fraud F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if two_classes else np.nan,
        "PR-AUC": average_precision_score(y_true, y_prob)
        if two_classes else np.nan,
    }

    return total_loss / max(total_seen, 1), metrics


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def plot_metrics(metrics_df, out_path):
    metric_names = [
        "Accuracy", "Precision", "Recall", "Fraud F1", "ROC-AUC", "PR-AUC"
    ]
    model_names = metrics_df["Model"].tolist()
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=SURFACE)
    style_axes(ax)

    for i, model_name in enumerate(model_names):
        values = metrics_df.loc[
            metrics_df["Model"] == model_name, metric_names
        ].values.flatten()

        positions = x - 0.4 + width / 2 + i * width
        bars = ax.bar(
            positions,
            values,
            width=width * 0.9,
            color=COLORS.get(model_name),
            label=model_name,
            zorder=3,
        )

        for rect, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK_SECONDARY,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color=INK_PRIMARY, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Federated test-set metrics",
        color=INK_PRIMARY,
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


# ---------------------------------------------------------
# Federated experiment
# ---------------------------------------------------------

def main(method="fedavg"):
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "cpu" if FORCE_CPU else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    graph, labels, train_ids, val_ids, test_ids = load_graph_and_splits()

    metadata = (list(graph.node_types), list(graph.edge_types))
    in_dims = {
        node_type: graph[node_type].x.size(-1)
        for node_type in graph.node_types
    }

    class_weights = get_class_weights(labels, train_ids, device)

    print("Partitioning graph into clients...")
    edge_partitions = make_edge_partitions(graph, NUM_CLIENTS)
    client_graphs, client_global_ids = build_client_graphs(
        graph, edge_partitions
    )

    train_local, val_local, test_local = allocate_provider_splits(
        client_global_ids, train_ids, val_ids, test_ids
    )

    client_train_loaders = [
        make_loader(client_graphs[i], train_local[i], shuffle=True)
        for i in range(NUM_CLIENTS)
    ]
    client_val_loaders = [
        make_loader(client_graphs[i], val_local[i], shuffle=False)
        for i in range(NUM_CLIENTS)
    ]
    client_test_loaders = [
        make_loader(client_graphs[i], test_local[i], shuffle=False)
        for i in range(NUM_CLIENTS)
    ]

    builders = {
        "HeteroSAGE": lambda: HeteroSAGE(
            metadata, in_dims, hidden=HIDDEN_DIM, dropout=DROPOUT
        ),
        "HeteroHGT": lambda: HeteroHGT(
            metadata, in_dims, hidden=HIDDEN_DIM,
            heads=HGT_HEADS, dropout=DROPOUT
        ),
        "HeteroRGCN": lambda: HeteroRGCN(
            metadata, in_dims, hidden=HIDDEN_DIM, dropout=DROPOUT
        ),
    }

    rows = []

    for model_name, build_model in builders.items():
        print(f"\n========== {model_name} ==========")
        global_model = build_model().to(device)

        for round_id in range(1, FEDERATED_ROUNDS + 1):
            local_states = []
            client_sizes = []

            for client_id in range(NUM_CLIENTS):
                loader = client_train_loaders[client_id]
                num_examples = len(train_local[client_id])

                if loader is None or num_examples == 0:
                    continue

                local_model = build_model().to(device)
                local_model.load_state_dict(global_model.state_dict())

                train_local_model(
                    local_model,
                    loader,
                    class_weights,
                    device,
                    epochs=LOCAL_EPOCHS,
                )

                local_states.append(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in local_model.state_dict().items()
                    }
                )
                client_sizes.append(num_examples)
                del local_model

            if not local_states:
                raise ValueError("No clients contributed a local model.")
            accumulator = {}
            if method=="fedavg":
                global_state = fedavg(local_states, client_sizes)
            elif method=="fedadagrad":
                global_state, accumulator = fedadagrad(
                    global_model.state_dict(), local_states, client_sizes, accumulator
                )
            global_model.load_state_dict(global_state)

            val_loss, val_metrics = evaluate_model(
                global_model, client_val_loaders, class_weights, device
            )

            print(
                f"Round {round_id:02d}/{FEDERATED_ROUNDS} | "
                f"val_loss={val_loss:.4f} | "
                f"val_F1={val_metrics['Fraud F1']:.4f}"
            )

            free_memory()

        # Evaluate only after all federated rounds.
        test_loss, test_metrics = evaluate_model(
            global_model, client_test_loaders, class_weights, device
        )

        rows.append({
            "Model": model_name,
            **test_metrics,
            "TestLoss": test_loss,
            "FederatedRounds": FEDERATED_ROUNDS,
            "LocalEpochs": LOCAL_EPOCHS,
            "Clients": NUM_CLIENTS,
            "Params": sum(p.numel() for p in global_model.parameters()),
        })

        del global_model
        free_memory()

    metrics_df = pd.DataFrame(rows)
    print("\nFEDERATED TEST METRICS")
    print(metrics_df.round(4).to_string(index=False))

    metrics_path = os.path.join(OUTPUT_DIR, method + "_federated_metrics.csv")
    metrics_df.round(4).to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)

    plot_metrics(
        metrics_df,
        os.path.join(OUTPUT_DIR, method+"_federated_metrics.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated GNN Comparison")
    parser.add_argument(
        "--method",
        type=str,
        choices=["fedavg", "fedadagrad"],
        default="fedavg",
        help="Federated aggregation method (default: fedavg)",
    )
    args = parser.parse_args()
    main(args.method)
