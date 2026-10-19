"""Example: compare LINE and HOPE provider-fraud classification."""

import pandas as pd
import torch

from embeddings import (
    build_homogeneous_adjacency,
    evaluate_provider_embeddings,
    fit_hope_embeddings,
    fit_line_embeddings,
)


def main():
    graph = torch.load(
        "./datasets/graph/hetero_data.pt",
        map_location="cpu",
        weights_only=False,
    )

    adjacency, node_offsets, node_counts = build_homogeneous_adjacency(graph)
    print(
        f"Homogeneous graph: {adjacency.shape[0]:,} nodes, "
        f"{adjacency.nnz // 2:,} undirected edges; "
        f"providers: {node_counts['provider']:,}"
    )

    line_embeddings = fit_line_embeddings(graph)
    hope_embeddings = fit_hope_embeddings(graph)

    line_result = evaluate_provider_embeddings(
        line_embeddings, graph, node_offsets, node_counts, method_name="LINE"
    )
    hope_result = evaluate_provider_embeddings(
        hope_embeddings, graph, node_offsets, node_counts,
        method_name="HOPE (Katz)"
    )

    results = pd.DataFrame([line_result, hope_result])
    print("\nProvider fraud classification from topology embeddings (test set)")
    print(results.to_string(index=False, formatters={
        "Accuracy": "{:.4f}".format,
        "Fraud F1": "{:.4f}".format,
        "ROC-AUC": "{:.4f}".format,
    }))


if __name__ == "__main__":
    main()
