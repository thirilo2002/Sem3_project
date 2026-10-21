"""Example: compare LINE and HOPE provider-fraud classification."""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.embeddings import (
    build_homogeneous_adjacency,
    evaluate_provider_embeddings,
    fit_hope_embeddings,
    fit_line_embeddings,
)


# ...existing code...
from pathlib import Path

import matplotlib.pyplot as plt
# ...existing code...


def plot_embedding_metrics(results, save_path=None, show=True):
    """Plot Accuracy, Precision, Recall, Fraud F1, ROC-AUC, and PR-AUC."""
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Fraud F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    methods = [result["Method"] for result in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, result in enumerate(results):
        values = [result.get(metric, np.nan) for metric in metrics]
        offset = (i - (len(results) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=methods[i])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    ax.set_title("Provider fraud test-set metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

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

    # ...existing code...
    results = [line_result, hope_result]

    plot_embedding_metrics(
        results,
        save_path="data/graph_visualizations/line_hope_metrics.png",
        show=False,
    )
# ...existing code...

    
    # print(f"\nComparison plot saved to: {output_path}")
    # print("\nProvider fraud classification from topology embeddings (test set)")
    # print(results.to_string(index=False, formatters={
    #     "Accuracy": "{:.4f}".format,
    #     "Fraud F1": "{:.4f}".format,
    #     "ROC-AUC": "{:.4f}".format,
    # }))


if __name__ == "__main__":
    main()
