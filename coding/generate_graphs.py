# Assuming 'data' is your existing PyG HeteroData object
from models.graph_ddpm.ddpm import HealthcareGraphDDPM
from models.graph_ddpm.config_graph import (GRAPH_PATH,OUTPUT_DIR)
from models.graph_ddpm.training import load_graph_and_splits
import torch

data = torch.load(GRAPH_PATH)  # Load your graph data
ddpm = HealthcareGraphDDPM(data, hidden_dim=128, timesteps=1000)

# Run a training epoch loop on your graph / sampled subgraphs
for epoch in range(10): # Increase epochs for convergence
    loss = ddpm.train_step(data)
    print(f"Epoch {epoch} | Loss: {loss:.4f}")