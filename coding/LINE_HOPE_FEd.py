"""Federated LINE-style/HOPE-style embeddings and provider classifiers.

The heterogeneous graph is converted to homogeneous topology for embedding
training. Embeddings and classifiers use global model states aggregated with
FedAvg or FedAdagrad. This is an experimental adaptation, not an exact
implementation of the original centralized LINE or HOPE algorithms.
"""

import argparse
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, eye

from models.gnn_models.config import GRAPH_PATH, OUTPUT_DIR, SEED
from run_federated_learning import make_edge_partitions


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_CLIENTS = 5

# Keep these small for a quick test; increase them for experiments.
FEDERATED_ROUNDS = 3
LOCAL_EPOCHS = 1
CLASSIFIER_LOCAL_EPOCHS = 3
EMBEDDING_DIM = 64
BATCH_SIZE = 4096
NEGATIVE_SAMPLES = 3
LEARNING_RATE = 0.01
CLASSIFIER_LEARNING_RATE = 0.01

# HOPE-style truncated Katz proximity settings.
KATZ_STEPS = 2
KATZ_ALPHA = 0.01
MAX_HOPE_PAIRS = 10_000

TEST_SIZE = 0.20
METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "Fraud F1",
    "ROC-AUC",
    "PR-AUC",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Graph partitioning and global node IDs
# ---------------------------------------------------------

def make_client_edges(graph, num_clients=NUM_CLIENTS, seed=SEED):
    """Partition heterogeneous edges and convert endpoints to global IDs."""
    edge_partitions = make_edge_partitions(
        graph,
        num_clients=num_clients,
        seed=seed,
    )

    node_offsets = {}
    total_nodes = 0
    for node_type in graph.node_types:
        node_offsets[node_type] = total_nodes
        total_nodes += int(graph[node_type].num_nodes)

    client_edges = []
    client_nodes = []

    for client_id in range(num_clients):
        edge_parts = []
        nodes = set()

        for edge_type in graph.edge_types:
            selected_ids = edge_partitions[client_id][edge_type]
            if not selected_ids:
                continue

            src_type, _, dst_type = edge_type
            edge_ids = torch.tensor(selected_ids, dtype=torch.long)
            edge_index = graph[edge_type].edge_index[:, edge_ids]

            global_src = edge_index[0] + node_offsets[src_type]
            global_dst = edge_index[1] + node_offsets[dst_type]

            edge_parts.append(torch.stack((global_src, global_dst)))
            nodes.update(global_src.tolist())
            nodes.update(global_dst.tolist())

        if edge_parts:
            edges = torch.cat(edge_parts, dim=1)
        else:
            edges = torch.empty((2, 0), dtype=torch.long)

        client_edges.append(edges)
        client_nodes.append(nodes)

    # Assign isolated nodes to one client so every graph node has an owner.
    assigned_nodes = set().union(*client_nodes)
    missing_nodes = sorted(set(range(total_nodes)) - assigned_nodes)
    for index, node_id in enumerate(missing_nodes):
        client_nodes[index % num_clients].add(node_id)

    return client_edges, client_nodes, node_offsets, total_nodes


def make_undirected_unique_edges(edge_index):
    """Make edges undirected and remove duplicate pairs."""
    if edge_index.numel() == 0:
        return edge_index

    edges = edge_index.t().cpu().numpy()
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    edges = np.unique(edges, axis=0)
    return torch.as_tensor(edges.T.copy(), dtype=torch.long)


# ---------------------------------------------------------
# Embedding initialization and aggregation
# ---------------------------------------------------------

def initialize_embedding_state(num_nodes, dimensions, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scale = 0.5 / max(dimensions, 1)

    return {
        "source": (
            torch.randn(num_nodes, dimensions, generator=generator) * scale
        ).to(device),
        "context": (
            torch.randn(num_nodes, dimensions, generator=generator) * scale
        ).to(device),
    }


def aggregate_embedding_fedavg(global_state, local_results):
    """FedAvg embedding rows, weighted by each client's row usage."""
    new_state = copy.deepcopy(global_state)

    for key, current in global_state.items():
        weighted_delta = torch.zeros_like(current)
        row_weights = torch.zeros(
            current.size(0),
            dtype=current.dtype,
            device=current.device,
        )

        for local_state, usage in local_results:
            counts = usage[key].to(
                device=current.device,
                dtype=current.dtype,
            )
            weighted_delta += (
                local_state[key].to(current.device) - current
            ) * counts[:, None]
            row_weights += counts

        active = row_weights > 0
        new_state[key][active] = (
            current[active]
            + weighted_delta[active] / row_weights[active, None]
        )

    return new_state


def aggregate_embedding_fedadagrad(
    global_state,
    local_results,
    accumulator,
    learning_rate=0.01,
    tau=1e-8,
):
    """Apply row-wise FedAdagrad; preserve the accumulator across rounds."""
    averaged_state = aggregate_embedding_fedavg(global_state, local_results)
    new_state = copy.deepcopy(global_state)

    for key, current in global_state.items():
        delta = averaged_state[key] - current
        if key not in accumulator:
            accumulator[key] = torch.zeros_like(current)

        accumulator[key].add_(delta.square())
        new_state[key] = current + learning_rate * delta / (
            accumulator[key].sqrt() + tau
        )

    return new_state, accumulator


def aggregate_classifier_fedavg(global_state, local_states, client_sizes):
    """Weighted FedAvg for classifier parameters."""
    total_size = sum(client_sizes)
    if total_size <= 0:
        raise ValueError("No clients have classifier training examples.")

    return {
        key: sum(
            state[key].to(global_state[key].device) * (size / total_size)
            for state, size in zip(local_states, client_sizes)
        )
        for key in global_state
    }


def aggregate_classifier_fedadagrad(
    global_state,
    local_states,
    client_sizes,
    accumulator,
    learning_rate=0.01,
    tau=1e-8,
):
    """Weighted FedAdagrad for classifier parameters."""
    total_size = sum(client_sizes)
    if total_size <= 0:
        raise ValueError("No clients have classifier training examples.")

    new_state = copy.deepcopy(global_state)

    for key, current in global_state.items():
        average_local = sum(
            state[key].to(current.device) * (size / total_size)
            for state, size in zip(local_states, client_sizes)
        )
        delta = average_local - current

        if key not in accumulator:
            accumulator[key] = torch.zeros_like(delta)

        accumulator[key].add_(delta.square())
        new_state[key] = current + learning_rate * delta / (
            accumulator[key].sqrt() + tau
        )

    return new_state, accumulator


# ---------------------------------------------------------
# Federated LINE-style embedding training
# ---------------------------------------------------------

def train_line_client(global_state, edges, client_nodes, device, seed):
    """Train a local negative-sampling, LINE-style objective."""
    if edges.size(1) == 0 or not client_nodes:
        return None

    set_seed(seed)

    source = nn.Embedding.from_pretrained(
        global_state["source"].detach().clone(),
        freeze=False,
    ).to(device)
    context = nn.Embedding.from_pretrained(
        global_state["context"].detach().clone(),
        freeze=False,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(source.parameters()) + list(context.parameters()),
        lr=LEARNING_RATE,
    )

    edge_tensor = edges.to(device)
    node_pool = torch.tensor(
        sorted(client_nodes),
        dtype=torch.long,
        device=device,
    )

    usage = {
        "source": torch.zeros(source.num_embeddings),
        "context": torch.zeros(context.num_embeddings),
    }

    for _ in range(LOCAL_EPOCHS):
        order = torch.randperm(edge_tensor.size(1), device=device)

        for batch_ids in order.split(BATCH_SIZE):
            batch = edge_tensor[:, batch_ids]
            src, dst = batch[0], batch[1]
            batch_size = src.numel()

            negative_ids = node_pool[
                torch.randint(
                    node_pool.numel(),
                    (batch_size, NEGATIVE_SAMPLES),
                    device=device,
                )
            ]

            positive_score = (source(src) * context(dst)).sum(dim=-1)
            negative_score = (
                source(src).unsqueeze(1) * context(negative_ids)
            ).sum(dim=-1)

            loss = -F.logsigmoid(positive_score).mean()
            loss -= F.logsigmoid(-negative_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            usage["source"].index_add_(
                0, src.cpu(), torch.ones(src.numel())
            )
            usage["context"].index_add_(
                0, dst.cpu(), torch.ones(dst.numel())
            )
            usage["context"].index_add_(
                0,
                negative_ids.reshape(-1).cpu(),
                torch.ones(negative_ids.numel()),
            )

    local_state = {
        "source": source.weight.detach().cpu(),
        "context": context.weight.detach().cpu(),
    }
    return local_state, usage


# ---------------------------------------------------------
# Federated HOPE-style embedding training
# ---------------------------------------------------------

def make_katz_pairs(edges, client_nodes, seed):
    """Build sampled pairs from a truncated, client-local Katz matrix."""
    if edges.size(1) == 0 or len(client_nodes) < 2:
        return None

    node_ids = np.asarray(sorted(client_nodes), dtype=np.int64)
    global_to_local = {
        int(global_id): local_id
        for local_id, global_id in enumerate(node_ids)
    }

    edge_array = edges.cpu().numpy()
    local_src = np.fromiter(
        (global_to_local[int(node)] for node in edge_array[0]),
        dtype=np.int64,
        count=edge_array.shape[1],
    )
    local_dst = np.fromiter(
        (global_to_local[int(node)] for node in edge_array[1]),
        dtype=np.int64,
        count=edge_array.shape[1],
    )

    adjacency = coo_matrix(
        (
            np.ones(local_src.size, dtype=np.float32),
            (local_src, local_dst),
        ),
        shape=(node_ids.size, node_ids.size),
    ).tocsr()

    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Truncated Katz: alpha*A + alpha^2*A^2 + ...
    power = adjacency.copy()
    katz = adjacency.multiply(KATZ_ALPHA).tocsr()

    for step in range(2, KATZ_STEPS + 1):
        power = (power @ adjacency).tocsr()
        katz = katz + power.multiply(KATZ_ALPHA ** step)

    katz = katz.tocoo()
    if katz.nnz == 0:
        return None

    rows = katz.row
    cols = katz.col
    values = katz.data.astype(np.float32)

    rng = np.random.default_rng(seed)
    if values.size > MAX_HOPE_PAIRS:
        selected = rng.choice(
            values.size,
            size=MAX_HOPE_PAIRS,
            replace=False,
        )
        rows = rows[selected]
        cols = cols[selected]
        values = values[selected]

    values = values / max(float(values.max()), 1e-12)

    # Add sampled non-neighbor pairs with zero target proximity.
    positive_keys = set(
        (rows.astype(np.int64) * node_ids.size + cols.astype(np.int64)).tolist()
    )
    zero_rows = []
    zero_cols = []
    target_zeros = len(rows)
    attempts = 0

    while len(zero_rows) < target_zeros and attempts < target_zeros * 20:
        row = int(rng.integers(node_ids.size))
        col = int(rng.integers(node_ids.size))
        key = row * node_ids.size + col

        if row != col and key not in positive_keys:
            zero_rows.append(row)
            zero_cols.append(col)
        attempts += 1

    local_rows = np.concatenate((rows, np.asarray(zero_rows, dtype=np.int64)))
    local_cols = np.concatenate((cols, np.asarray(zero_cols, dtype=np.int64)))
    targets = np.concatenate((
        values,
        np.zeros(len(zero_rows), dtype=np.float32),
    ))

    return (
        torch.as_tensor(node_ids[local_rows], dtype=torch.long),
        torch.as_tensor(node_ids[local_cols], dtype=torch.long),
        torch.as_tensor(targets, dtype=torch.float32),
    )


def train_hope_client(global_state, pairs, device):
    """Factorize local Katz proximity pairs with a squared-error objective."""
    if pairs is None:
        return None

    pair_src, pair_dst, targets = pairs

    source = nn.Embedding.from_pretrained(
        global_state["source"].detach().clone(),
        freeze=False,
    ).to(device)
    context = nn.Embedding.from_pretrained(
        global_state["context"].detach().clone(),
        freeze=False,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(source.parameters()) + list(context.parameters()),
        lr=LEARNING_RATE,
    )

    pair_src = pair_src.to(device)
    pair_dst = pair_dst.to(device)
    targets = targets.to(device)

    usage = {
        "source": torch.zeros(source.num_embeddings),
        "context": torch.zeros(context.num_embeddings),
    }

    for _ in range(LOCAL_EPOCHS):
        order = torch.randperm(pair_src.numel(), device=device)

        for batch_ids in order.split(BATCH_SIZE):
            src = pair_src[batch_ids]
            dst = pair_dst[batch_ids]
            target = targets[batch_ids]

            prediction = (source(src) * context(dst)).sum(dim=-1)
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            usage["source"].index_add_(
                0, src.cpu(), torch.ones(src.numel())
            )
            usage["context"].index_add_(
                0, dst.cpu(), torch.ones(dst.numel())
            )

    local_state = {
        "source": source.weight.detach().cpu(),
        "context": context.weight.detach().cpu(),
    }
    return local_state, usage


def train_federated_embeddings(
    embedding_method,
    aggregation,
    client_edges,
    client_nodes,
    num_nodes,
    device,
):
    """Train and aggregate one embedding model."""
    method_seed = SEED + (0 if embedding_method == "LINE" else 10_000)
    global_state = initialize_embedding_state(
        num_nodes,
        EMBEDDING_DIM,
        method_seed,
        device,
    )
    accumulator = {}

    hope_pairs = None
    if embedding_method == "HOPE":
        hope_pairs = [
            make_katz_pairs(edges, nodes, SEED + client_id)
            for client_id, (edges, nodes) in enumerate(
                zip(client_edges, client_nodes)
            )
        ]

    for round_id in range(1, FEDERATED_ROUNDS + 1):
        local_results = []

        for client_id in range(len(client_edges)):
            if embedding_method == "LINE":
                result = train_line_client(
                    global_state,
                    client_edges[client_id],
                    client_nodes[client_id],
                    device,
                    seed=SEED + round_id * 100 + client_id,
                )
            else:
                result = train_hope_client(
                    global_state,
                    hope_pairs[client_id],
                    device,
                )

            if result is not None:
                local_results.append(result)

        if not local_results:
            raise ValueError(
                f"No clients produced updates for {embedding_method}."
            )

        if aggregation == "fedavg":
            global_state = aggregate_embedding_fedavg(
                global_state, local_results
            )
        elif aggregation == "fedadagrad":
            global_state, accumulator = aggregate_embedding_fedadagrad(
                global_state,
                local_results,
                accumulator,
            )
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        print(
            f"{embedding_method} / {aggregation}: "
            f"round {round_id}/{FEDERATED_ROUNDS}, "
            f"clients={len(local_results)}"
        )

    embeddings = (
        (global_state["source"] + global_state["context"]) * 0.5
    ).detach().cpu()

    return embeddings


# ---------------------------------------------------------
# Provider classifier split and ownership
# ---------------------------------------------------------

def make_provider_split_and_owners(
    graph,
    client_nodes,
    node_offsets,
    test_size=TEST_SIZE,
    seed=SEED,
):
    """Create a stratified provider split and assign providers to clients."""
    from sklearn.model_selection import train_test_split

    labels = graph["provider"].y.long().view(-1).cpu().numpy()
    provider_ids = np.arange(len(labels))

    train_ids, test_ids = train_test_split(
        provider_ids,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    provider_offset = node_offsets["provider"]
    provider_end = provider_offset + len(labels)

    owner = {}
    for client_id, global_ids in enumerate(client_nodes):
        for global_id in global_ids:
            if provider_offset <= global_id < provider_end:
                owner.setdefault(int(global_id), client_id)

    train_by_client = [[] for _ in client_nodes]
    test_by_client = [[] for _ in client_nodes]

    for local_provider_id in train_ids:
        global_id = provider_offset + int(local_provider_id)
        if global_id not in owner:
            raise ValueError(f"Provider node {global_id} has no client owner.")
        train_by_client[owner[global_id]].append(int(local_provider_id))

    for local_provider_id in test_ids:
        global_id = provider_offset + int(local_provider_id)
        if global_id not in owner:
            raise ValueError(f"Provider node {global_id} has no client owner.")
        test_by_client[owner[global_id]].append(int(local_provider_id))

    return labels, train_ids, test_ids, train_by_client, test_by_client


# ---------------------------------------------------------
# Federated logistic regression / linear SVM
# ---------------------------------------------------------

def train_federated_classifier(
    embeddings,
    graph,
    client_nodes,
    node_offsets,
    embedding_method,
    aggregation,
    classifier_kind,
    device,
):
    """Train classifier weights locally and aggregate them at the server."""
    labels, train_ids, _, train_by_client, test_by_client = (
        make_provider_split_and_owners(
            graph,
            client_nodes,
            node_offsets,
        )
    )

    if isinstance(embeddings, torch.Tensor):
        embedding_matrix = embeddings.detach().cpu().float()
    else:
        embedding_matrix = torch.as_tensor(
            embeddings,
            dtype=torch.float32,
        )

    provider_offset = node_offsets["provider"]
    feature_dim = embedding_matrix.size(1)

    model = nn.Linear(feature_dim, 1).to(device)
    global_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    accumulator = {}

    for round_id in range(1, FEDERATED_ROUNDS + 1):
        local_states = []
        client_sizes = []

        for local_provider_ids in train_by_client:
            if not local_provider_ids:
                continue

            ids = np.asarray(local_provider_ids, dtype=np.int64)
            global_ids = provider_offset + ids

            x = embedding_matrix[global_ids].to(device)
            y = torch.as_tensor(
                labels[ids],
                dtype=torch.float32,
                device=device,
            )

            local_model = nn.Linear(feature_dim, 1).to(device)
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=CLASSIFIER_LEARNING_RATE,
                weight_decay=1e-4,
            )

            local_model.train()
            for _ in range(CLASSIFIER_LOCAL_EPOCHS):
                logits = local_model(x).squeeze(-1)

                if classifier_kind == "logistic":
                    loss = F.binary_cross_entropy_with_logits(logits, y)
                elif classifier_kind == "svm":
                    signed_labels = 2 * y - 1
                    loss = F.relu(1 - signed_labels * logits).mean()
                else:
                    raise ValueError(
                        f"Unknown classifier: {classifier_kind}"
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            local_states.append({
                key: value.detach().cpu().clone()
                for key, value in local_model.state_dict().items()
            })
            client_sizes.append(len(ids))

        if not local_states:
            raise ValueError("No clients trained the provider classifier.")

        if aggregation == "fedavg":
            global_state = aggregate_classifier_fedavg(
                global_state,
                local_states,
                client_sizes,
            )
        elif aggregation == "fedadagrad":
            global_state, accumulator = aggregate_classifier_fedadagrad(
                global_state,
                local_states,
                client_sizes,
                accumulator,
            )
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        print(
            f"{classifier_kind} classifier / {aggregation}: "
            f"round {round_id}/{FEDERATED_ROUNDS}"
        )

    model.load_state_dict(global_state)
    model.eval()

    y_true_parts = []
    score_parts = []

    with torch.no_grad():
        for local_provider_ids in test_by_client:
            if not local_provider_ids:
                continue

            ids = np.asarray(local_provider_ids, dtype=np.int64)
            global_ids = provider_offset + ids
            x = embedding_matrix[global_ids].to(device)

            scores = model(x).squeeze(-1).cpu().numpy()
            y_true_parts.extend(labels[ids].tolist())
            score_parts.extend(scores.tolist())

    y_true = np.asarray(y_true_parts, dtype=np.int64)
    scores = np.asarray(score_parts, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
    predictions = (scores >= 0).astype(np.int64)
    has_both_classes = len(np.unique(y_true)) == 2

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "Method": f"{embedding_method}-{aggregation}-{classifier_kind}",
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "Fraud F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": (
            roc_auc_score(y_true, scores)
            if has_both_classes
            else np.nan
        ),
        "PR-AUC": (
            average_precision_score(y_true, scores)
            if has_both_classes
            else np.nan
        ),
    }


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def plot_metrics(results, output_path, title):
    """Save a grouped plot for the six provider test metrics."""
    x = np.arange(len(METRICS))
    width = 0.8 / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(13, 6))

    for index, result in enumerate(results):
        values = [result.get(metric, np.nan) for metric in METRICS]
        offset = (index - (len(results) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.9,
            label=result["Method"],
        )

        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main(aggregation_choice, classifier_choice):
    set_seed(SEED)

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if "y" not in graph["provider"]:
        raise ValueError("Graph must contain provider labels at graph['provider'].y")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    client_edges, client_nodes, node_offsets, num_nodes = make_client_edges(
        graph,
        num_clients=NUM_CLIENTS,
        seed=SEED,
    )

    aggregations = (
        ["fedavg", "fedadagrad"]
        if aggregation_choice == "both"
        else [aggregation_choice]
    )
    classifiers = (
        ["logistic", "svm"]
        if classifier_choice == "both"
        else [classifier_choice]
    )

    results = []

    for aggregation in aggregations:
        for embedding_method in ("LINE", "HOPE"):
            print(f"\n=== Train {embedding_method} with {aggregation} ===")
            embeddings = train_federated_embeddings(
                embedding_method=embedding_method,
                aggregation=aggregation,
                client_edges=client_edges,
                client_nodes=client_nodes,
                num_nodes=num_nodes,
                device=device,
            )

            for classifier_kind in classifiers:
                print(
                    f"\n=== Train federated {classifier_kind} classifier "
                    f"on {embedding_method} embeddings ({aggregation}) ==="
                )
                result = train_federated_classifier(
                    embeddings=embeddings,
                    graph=graph,
                    client_nodes=client_nodes,
                    node_offsets=node_offsets,
                    embedding_method=embedding_method,
                    aggregation=aggregation,
                    classifier_kind=classifier_kind,
                    device=device,
                )
                results.append(result)

    results_df = pd.DataFrame(results)
    print("\nFederated provider test metrics")
    print(
        results_df.to_string(
            index=False,
            formatters={
                metric: "{:.4f}".format
                for metric in METRICS
                if metric in results_df.columns
            },
        )
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{aggregation_choice}_{classifier_choice}"
    csv_path = output_dir / f"line_hope_federated_{suffix}.csv"
    plot_path = output_dir / f"line_hope_federated_{suffix}.png"

    results_df.to_csv(csv_path, index=False)
    plot_metrics(
        results,
        plot_path,
        title=f"Federated LINE/HOPE provider test metrics ({suffix})",
    )

    print(f"\nSaved metrics: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated LINE/HOPE with federated provider classifiers"
    )
    parser.add_argument(
        "--aggregation",
        choices=["fedavg", "fedadagrad", "both"],
        default="both",
        help="Embedding and classifier aggregation method(s).",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic", "svm", "both"],
        default="both",
        help="Federated provider classifier(s) to train.",
    )
    args = parser.parse_args()

    main(
        aggregation_choice=args.aggregation,
        classifier_choice=args.classifier,
    )