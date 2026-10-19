"""Run the HeteroSAGE, HeteroHGT, and HeteroRGCN comparison."""

import gc
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from models.gnn_models.config import (
    SEED, OUTPUT_DIR, HIDDEN_DIM, DROPOUT, HGT_HEADS, EPOCHS,
    BATCH_SIZE, NUM_NEIGHBORS, FORCE_CPU, COLORS, INK_PRIMARY,
    INK_SECONDARY, GRIDLINE, BASELINE, SURFACE,
)
from models.gnn_models.models import HeteroSAGE, HeteroHGT, HeteroRGCN
from models.gnn_models.training import (
    load_graph_and_splits, make_neighbor_loader, train_and_evaluate,
)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def plot_training_curves(histories, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    panels = [
        ("train_loss", "Training loss", "Weighted cross-entropy"),
        ("val_loss", "Validation loss", "Weighted cross-entropy"),
        ("val_f1", "Validation fraud F1", "F1 (fraud class)"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        style_axes(ax)
        for model_name, history in histories.items():
            ax.plot(
                history["epoch"], history[key], color=COLORS[model_name],
                linewidth=2, label=model_name, solid_capstyle="round",
            )
        ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel("Epoch", color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    axes[-1].legend(frameon=False, loc="lower right", labelcolor=INK_PRIMARY, fontsize=9)
    fig.suptitle(
        "Heterogeneous GNN comparison — Step 03 provider-fraud graph",
        color=INK_PRIMARY, fontsize=15, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def plot_metrics_comparison(metrics_df, out_path):
    metric_names = ["Accuracy", "Precision", "Recall", "Fraud F1", "ROC-AUC", "PR-AUC"]
    model_names = metrics_df["Model"].tolist()
    x = __import__("numpy").arange(len(metric_names))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=SURFACE)
    style_axes(ax)
    for i, model_name in enumerate(model_names):
        values = metrics_df.loc[metrics_df["Model"] == model_name, metric_names].values.flatten()
        positions = x - 0.4 + width / 2 + i * width
        bars = ax.bar(positions, values, width=width * 0.9, color=COLORS[model_name], label=model_name, zorder=3)
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                f"{value:.2f}", ha="center", va="bottom", fontsize=7.5,
                color=INK_SECONDARY,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color=INK_PRIMARY, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Test-set metrics by model", color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_PRIMARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(
        "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    graph, labels, train_ids, val_ids, test_ids = load_graph_and_splits()
    in_dims = {nt: graph[nt].x.size(-1) for nt in graph.node_types}
    metadata = (list(graph.node_types), list(graph.edge_types))
    class_counts = torch.bincount(labels[train_ids], minlength=2).float()
    class_weights = (class_counts.sum() / (2 * class_counts.clamp(min=1))).to(device)

    train_loader = make_neighbor_loader(graph, train_ids, shuffle=True)
    val_loader = make_neighbor_loader(graph, val_ids, shuffle=False)
    test_loader = make_neighbor_loader(graph, test_ids, shuffle=False)
    print(
        f"Device: {device}; batches: train={len(train_loader)} "
        f"val={len(val_loader)} test={len(test_loader)} "
        f"(batch_size={BATCH_SIZE}, neighbors={NUM_NEIGHBORS})"
    )

    builders = {
        "HeteroSAGE": lambda: HeteroSAGE(metadata, in_dims, hidden=HIDDEN_DIM, dropout=DROPOUT),
        "HeteroHGT": lambda: HeteroHGT(metadata, in_dims, hidden=HIDDEN_DIM, heads=HGT_HEADS, dropout=DROPOUT),
        "HeteroRGCN": lambda: HeteroRGCN(metadata, in_dims, hidden=HIDDEN_DIM, dropout=DROPOUT),
    }

    histories, rows = {}, []
    for model_name, build in builders.items():
        torch.manual_seed(SEED)
        model = build().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"\n--- {model_name} ({n_params:,} parameters) ---")
        history, test_metrics, best_epoch, train_seconds = train_and_evaluate(
            model_name, model, train_loader, val_loader, test_loader,
            class_weights, epochs=EPOCHS, device=device,
        )
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        histories[model_name] = history
        rows.append({
            "Model": model_name, **test_metrics, "BestEpoch": best_epoch,
            "Params": n_params, "TrainSeconds": round(train_seconds, 1),
            "PeakGPU_GB": round(peak_gb, 2),
        })
        del model
        free_memory()

    metrics_df = pd.DataFrame(rows)
    print("\nTEST-SET METRICS COMPARISON")
    print(metrics_df.round(4).to_string(index=False))
    metrics_df.round(4).to_csv(os.path.join(OUTPUT_DIR, "metrics_comparison.csv"), index=False)

    history_rows = []
    for model_name, history in histories.items():
        for epoch, train_loss, val_loss, val_f1 in zip(
            history["epoch"], history["train_loss"], history["val_loss"], history["val_f1"]
        ):
            history_rows.append({
                "Model": model_name, "Epoch": epoch, "TrainLoss": train_loss,
                "ValLoss": val_loss, "ValF1": val_f1,
            })
    pd.DataFrame(history_rows).to_csv(os.path.join(OUTPUT_DIR, "history.csv"), index=False)
    plot_training_curves(histories, os.path.join(OUTPUT_DIR, "training_curves.png"))
    plot_metrics_comparison(metrics_df, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))


if __name__ == "__main__":
    main()
