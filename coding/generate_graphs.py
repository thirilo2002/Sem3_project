# Assuming 'data' is your existing PyG HeteroData object
from models.graph_ddpm.ddpm import HealthcareGraphDDPM
from models.graph_ddpm.config_graph import (GRAPH_PATH,OUTPUT_DIR)
from models.graph_ddpm.training import load_graph_and_splits
import torch
data = torch.load(
    GRAPH_PATH,
    map_location="cpu",
    weights_only=False
)
ddpm = HealthcareGraphDDPM(data, hidden_dim=128, timesteps=1000)

# Run a training epoch loop on your graph / sampled subgraphs
for epoch in range(100): # Increase epochs for convergence
    loss = ddpm.train_step(data)
    print(f"Epoch {epoch} | Loss: {loss:.4f}")



print("Node types:", data.node_types)
print("Edge types:", data.edge_types)

for node_type in data.node_types:
    store = data[node_type]
    print(
        node_type,
        "nodes:", store.num_nodes,
        "features:", getattr(store, "x", None).shape
        if getattr(store, "x", None) is not None else None
    )

for edge_type in data.edge_types:
    print(edge_type, "edges:", data[edge_type].num_edges)