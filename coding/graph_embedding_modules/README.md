# Reusable LINE and HOPE modules

## Requirements

Install the packages used by the modules in your environment:

```bash
pip install numpy scipy scikit-learn torch pandas
```

Your graph should be a PyTorch Geometric `HeteroData`-like object with a
`provider` node type and `graph["provider"].y` labels for evaluation.

## Use each method independently

```python
import torch
from embeddings import fit_line_embeddings

graph = torch.load(
    "./datasets/graph/hetero_data.pt",
    map_location="cpu",
    weights_only=False,
)

line_embeddings = fit_line_embeddings(graph)
print(line_embeddings.shape)
```

```python
import torch
from embeddings import fit_hope_embeddings

graph = torch.load(
    "./datasets/graph/hetero_data.pt",
    map_location="cpu",
    weights_only=False,
)

hope_embeddings = fit_hope_embeddings(graph)
print(hope_embeddings.shape)
```

Both functions return NumPy arrays with shape `(total_nodes, embedding_dim)`.
Node rows follow the global ordering generated from `graph.node_types`.

## Change settings

```python
line_embeddings = fit_line_embeddings(
    graph,
    embedding_dim=32,
    steps=1500,
    batch_size=1024,
    negative_samples=5,
)

hope_embeddings = fit_hope_embeddings(graph, embedding_dim=32)
```

## Run the comparison script

Keep `main.py` and the `embeddings/` folder in the same project directory,
and run:

```bash
python main.py
```

The HOPE implementation is a truncated spectral approximation to Katz
proximity for an undirected graph; it avoids constructing the dense Katz matrix.
