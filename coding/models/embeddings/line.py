"""LINE graph embeddings using first- and second-order proximity."""

import numpy as np
import torch
import torch.nn.functional as F

from .utils import build_homogeneous_adjacency, SEED


def fit_line_embeddings(
    graph,
    embedding_dim=32,
    steps=1500,
    batch_size=1024,
    negative_samples=5,
    learning_rate=0.025,
    seed=SEED,
    verbose=True,
):
    """Train LINE embeddings and return a NumPy array shaped (total_nodes, embedding_dim).

    The output concatenates the first-order embedding with the second-order
    source embedding, matching the approach in the original script.
    """
    if embedding_dim < 2 or embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be an even integer >= 2.")

    adjacency, _, _ = build_homogeneous_adjacency(graph)
    total_nodes = adjacency.shape[0]
    edge_sources_np, edge_targets_np = adjacency.nonzero()

    if len(edge_sources_np) == 0:
        raise ValueError("The graph contains no edges to embed.")

    edge_sources = torch.from_numpy(edge_sources_np.astype(np.int64, copy=False))
    edge_targets = torch.from_numpy(edge_targets_np.astype(np.int64, copy=False))

    node_degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    negative_weights = torch.from_numpy(
        np.power(node_degrees, 0.75).astype(np.float64)
    )
    if negative_weights.sum() == 0:
        raise ValueError("Cannot sample negative nodes from a graph with no edges.")
    negative_probabilities = negative_weights / negative_weights.sum()

    generator = torch.Generator().manual_seed(seed)
    half_dim = embedding_dim // 2

    line_first = torch.nn.Embedding(total_nodes, half_dim, sparse=True)
    line_source = torch.nn.Embedding(total_nodes, half_dim, sparse=True)
    line_context = torch.nn.Embedding(total_nodes, half_dim, sparse=True)

    for embedding in (line_first, line_source, line_context):
        torch.nn.init.xavier_uniform_(embedding.weight)

    optimizer = torch.optim.SparseAdam(
        list(line_first.parameters())
        + list(line_source.parameters())
        + list(line_context.parameters()),
        lr=learning_rate,
    )

    for step in range(steps):
        edge_ids = torch.randint(
            len(edge_sources), (batch_size,), generator=generator
        )
        source_ids = edge_sources[edge_ids]
        target_ids = edge_targets[edge_ids]
        negative_ids = torch.multinomial(
            negative_probabilities,
            batch_size * negative_samples,
            replacement=True,
            generator=generator,
        ).reshape(batch_size, negative_samples)

        first_source = line_first(source_ids)
        first_positive = (first_source * line_first(target_ids)).sum(dim=1)
        first_negative = (
            first_source.unsqueeze(1) * line_first(negative_ids)
        ).sum(dim=2)

        second_source = line_source(source_ids)
        second_positive = (
            second_source * line_context(target_ids)
        ).sum(dim=1)
        second_negative = (
            second_source.unsqueeze(1) * line_context(negative_ids)
        ).sum(dim=2)

        loss = (
            -F.logsigmoid(first_positive).mean()
            -F.logsigmoid(-first_negative).mean()
            -F.logsigmoid(second_positive).mean()
            -F.logsigmoid(-second_negative).mean()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % 500 == 0:
            print(f"LINE step {step + 1}/{steps} | loss={loss.item():.4f}")

    embeddings = torch.cat(
        [line_first.weight.detach(), line_source.weight.detach()], dim=1
    )
    return embeddings.cpu().numpy().astype(np.float32, copy=False)
