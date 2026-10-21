"""Shared utilities for heterogeneous graph embedding experiments."""

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
SEED = 42


def build_homogeneous_adjacency(graph):
    """Convert a PyG heterogeneous graph to a binary, undirected CSR adjacency.

    Nodes from all node types are concatenated in ``graph.node_types`` order.
    Returns:
        adjacency: scipy.sparse.csr_matrix
        node_offsets: dict mapping node type to its starting global node ID
        node_counts: dict mapping node type to its node count
    """
    node_counts = {
        node_type: int(graph[node_type].num_nodes)
        for node_type in graph.node_types
    }

    node_offsets = {}
    total_nodes = 0
    for node_type, count in node_counts.items():
        node_offsets[node_type] = total_nodes
        total_nodes += count

    edge_rows = []
    edge_columns = []

    for edge_type in graph.edge_types:
        source_type, _, target_type = edge_type
        edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
        if edge_index.shape[1] == 0:
            continue

        edge_rows.append(edge_index[0] + node_offsets[source_type])
        edge_columns.append(edge_index[1] + node_offsets[target_type])

    if not edge_rows:
        raise ValueError("The heterogeneous graph contains no edges to embed.")

    rows = np.concatenate(edge_rows).astype(np.int64, copy=False)
    columns = np.concatenate(edge_columns).astype(np.int64, copy=False)

    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(total_nodes, total_nodes),
    ).tocsr()

    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data[:] = 1.0

    return adjacency, node_offsets, node_counts


def get_provider_split(graph, node_offsets, node_counts, test_size=0.20, seed=SEED):
    """Return provider labels, global provider IDs, and stratified train/test IDs."""
    if "provider" not in node_offsets:
        raise KeyError("Expected a node type named 'provider' in the graph.")

    provider_offset = node_offsets["provider"]
    provider_count = node_counts["provider"]
    provider_labels = (
        graph["provider"].y.detach().cpu().numpy().astype(np.int64)
    )
    provider_node_ids = np.arange(
        provider_offset, provider_offset + provider_count, dtype=np.int64
    )

    train_ids, test_ids = train_test_split(
        np.arange(provider_count),
        test_size=test_size,
        random_state=seed,
        stratify=provider_labels,
    )

    return provider_labels, provider_node_ids, train_ids, test_ids




def evaluate_provider_embeddings(
    embeddings,
    graph,
    node_offsets,
    node_counts,
    method_name="Embedding",
    test_size=0.20,
    seed=SEED,
    plot_path=None,
    show_plot=False
):
    """Train/evaluate a logistic-regression provider classifier from embeddings.

    Graph structure is used unsupervised to create embeddings. Only training
    provider labels are used to fit the classifier.
    """
    labels, provider_node_ids, train_ids, test_ids = get_provider_split(
        graph, node_offsets, node_counts, test_size=test_size, seed=seed
    )
    provider_features = embeddings[provider_node_ids]

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    classifier.fit(provider_features[train_ids], labels[train_ids])

    predictions = classifier.predict(provider_features[test_ids])
    probabilities = classifier.predict_proba(provider_features[test_ids])[:, 1]
    
    has_both_classes = len(np.unique(labels[test_ids])) == 2
    return {
        "Method": method_name,
        "Accuracy": accuracy_score(labels[test_ids], predictions),
        "Precision": precision_score(
            labels[test_ids], predictions, zero_division=0
        ),
        "Recall": recall_score(
            labels[test_ids], predictions, zero_division=0
        ),
        "Fraud F1": f1_score(
            labels[test_ids], predictions, zero_division=0
        ),
        "ROC-AUC": (
            roc_auc_score(labels[test_ids], probabilities)
            if has_both_classes
            else np.nan
        ),
        "PR-AUC": (
            average_precision_score(labels[test_ids], probabilities)
            if has_both_classes
            else np.nan
        ),
    }