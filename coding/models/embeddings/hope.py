"""HOPE-style embeddings based on a truncated Katz-proximity eigendecomposition."""

import numpy as np
from scipy.sparse.linalg import eigsh

from .utils import build_homogeneous_adjacency, SEED


def fit_hope_embeddings(
    graph,
    embedding_dim=32,
    seed=SEED,
    verbose=True,
):
    """Return HOPE/Katz embeddings as a NumPy array shaped (total_nodes, embedding_dim).

    For the undirected adjacency used here, this implementation approximates
    the Katz matrix through its eigenvectors instead of materializing a dense
    all-pairs Katz matrix. The result is a practical spectral approximation.
    """
    if embedding_dim < 2 or embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be an even integer >= 2.")

    adjacency, _, _ = build_homogeneous_adjacency(graph)
    total_nodes = adjacency.shape[0]
    half_dim = embedding_dim // 2

    if total_nodes <= 2:
        raise ValueError("HOPE requires at least three graph nodes.")

    if total_nodes <= half_dim + 1:
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency.toarray())
    else:
        eigenvalues, eigenvectors = eigsh(
            adjacency,
            k=half_dim,
            which="LM",
            v0=np.random.default_rng(seed).normal(size=total_nodes),
        )

    spectral_radius = np.max(np.abs(eigenvalues))
    if spectral_radius == 0:
        raise ValueError("HOPE cannot embed a graph with no nonzero eigenvalues.")

    katz_beta = 0.85 / spectral_radius
    katz_eigenvalues = eigenvalues / (1.0 - katz_beta * eigenvalues)
    rank_order = np.argsort(np.abs(katz_eigenvalues))[::-1][:half_dim]

    selected_values = katz_eigenvalues[rank_order]
    selected_vectors = eigenvectors[:, rank_order]
    scales = np.sqrt(np.abs(selected_values))

    hope_left = selected_vectors * scales
    hope_right = selected_vectors * (np.sign(selected_values) * scales)
    embeddings = np.concatenate([hope_left, hope_right], axis=1).astype(np.float32)

    if verbose:
        print(
            f"HOPE (Katz): generated {embeddings.shape[1]}-dimensional "
            f"embeddings for {embeddings.shape[0]:,} nodes."
        )

    return embeddings
