"""Configuration constants for heterogeneous GNN experiments."""

SEED = 42
GRAPH_PATH = "./datasets/graph/hetero_data.pt"
OUTPUT_DIR = "./outputs/step03_gnn_comparison"

HIDDEN_DIM = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
HGT_HEADS = 2
MAX_GRAD_NORM = 1.0

BATCH_SIZE = 256
NUM_NEIGHBORS = [10, 5]
FORCE_CPU = False

COLORS = {
    "HeteroSAGE": "#2a78d6",
    "HeteroHGT": "#eb6834",
    "HeteroRGCN": "#1baf7a",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
