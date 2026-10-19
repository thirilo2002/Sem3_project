"""Graph loading, feature preparation, mini-batch training, and evaluation."""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader

from .config_graph import (
    SEED, GRAPH_PATH, BATCH_SIZE, NUM_NEIGHBORS,
    LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM,
)


def signed_log1p(x):
    return torch.sign(x) * torch.log1p(x.abs())


def standardize_features(graph, train_provider_ids):
    """Signed-log transform and z-score features by node type.

    Provider statistics use training providers only; all nodes of other types
    are used because they have no supervised provider labels.
    """
    for node_type in graph.node_types:
        x = torch.nan_to_num(
            graph[node_type].x.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        x = signed_log1p(x)
        fit_rows = x[train_provider_ids] if node_type == "provider" else x
        mean = fit_rows.mean(dim=0, keepdim=True)
        std = fit_rows.std(dim=0, keepdim=True).clamp(min=1e-6)
        graph[node_type].x = (x - mean) / std


def load_graph_and_splits(graph_path=GRAPH_PATH, seed=SEED):
    """Load graph and return graph, labels, train IDs, validation IDs, test IDs."""
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    for node_type in graph.node_types:
        if "x" not in graph[node_type]:
            raise ValueError(f"Missing features for node type '{node_type}'")

    labels = graph["provider"].y.long().view(-1)
    num_providers = graph["provider"].num_nodes
    all_ids = np.arange(num_providers)

    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.30, random_state=seed, stratify=labels.numpy()
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=seed,
        stratify=labels.numpy()[temp_ids],
    )
    train_ids = torch.as_tensor(train_ids, dtype=torch.long)
    val_ids = torch.as_tensor(val_ids, dtype=torch.long)
    test_ids = torch.as_tensor(test_ids, dtype=torch.long)

    standardize_features(graph, train_ids)

    print(
        f"Providers: {num_providers} | train={len(train_ids)} "
        f"val={len(val_ids)} test={len(test_ids)}"
    )
    classes, counts = torch.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(classes.tolist(), counts.tolist())))
    return graph, labels, train_ids, val_ids, test_ids


def make_neighbor_loader(
    graph, ids, shuffle=False, batch_size=BATCH_SIZE,
    num_neighbors=None, num_workers=0
):
    """Create a provider-seeded NeighborLoader."""
    return NeighborLoader(
        graph,
        num_neighbors=NUM_NEIGHBORS if num_neighbors is None else num_neighbors,
        input_nodes=("provider", ids),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(np.int64)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Fraud F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if two_classes else np.nan,
        "PR-AUC": average_precision_score(y_true, y_prob) if two_classes else np.nan,
    }


def evaluate(model, loader, class_weights, device=None):
    """Evaluate weighted loss and metrics over provider seed nodes."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_loss, total_seen = 0.0, 0
    targets, probabilities = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seeds = batch["provider"].batch_size
            logits = model(batch)
            y = batch["provider"].y[:seeds]
            total_loss += F.cross_entropy(
                logits, y, weight=class_weights
            ).item() * seeds
            total_seen += seeds
            targets.append(y.cpu())
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu())

    if not targets:
        raise ValueError("Evaluation loader produced no batches.")
    y_true = torch.cat(targets).numpy()
    y_prob = torch.cat(probabilities).numpy()
    return total_loss / max(total_seen, 1), compute_metrics(y_true, y_prob)


def train_and_evaluate(
    model_name, model, train_loader, val_loader, test_loader, class_weights,
    epochs=None, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM, device=None, verbose=True
):
    """Train using validation fraud-F1 for checkpoint selection, then test.

    Returns (history, test_metrics, best_epoch, train_seconds).
    """
    from .config_graph import EPOCHS
    epochs = EPOCHS if epochs is None else epochs
    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1, best_epoch, best_state = -1.0, 0, None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_seen = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            seeds = batch["provider"].batch_size
            y = batch["provider"].y[:seeds]
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * seeds
            total_seen += seeds

        train_loss = total_loss / max(total_seen, 1)
        val_loss, val_metrics = evaluate(
            model, val_loader, class_weights, device=device
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["Fraud F1"])

        if val_metrics["Fraud F1"] > best_val_f1:
            best_val_f1 = val_metrics["Fraud F1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"[{model_name}] epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_F1={val_metrics['Fraud F1']:.4f}"
            )

    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_metrics = evaluate(model, test_loader, class_weights, device=device)
    return history, test_metrics, best_epoch, train_seconds
