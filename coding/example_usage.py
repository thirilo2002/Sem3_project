"""
End-to-end example. Replace `load_real_heterodata()` with however you
already load your claims HeteroData (torch.load, a build script, etc.) --
it must have:
  - data[ntype].x for ntype in {beneficiary, claim, provider, physician,
    diagnosis, procedure}, with the feat_dims from schema.FEAT_DIMS
  - data[src, rel, dst].edge_index for every relation in schema.py
    (both forward and reverse directions, as in your original relations list)

Then:
    python example_usage.py
"""
import torch
from torch_geometric.data import HeteroData

from models.graphgen.schema import build_default_schema
from models.graphgen.train import fit_all
from models.graphgen.generate import generate_graph


def load_real_heterodata() -> HeteroData:


    return torch.load(
        "./datasets/graph/hetero_data.pt",
        map_location="cpu",
        weights_only=False
    )
 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schema = build_default_schema()  # EDIT schema.py's ColumnSpec blocks first!

    real_data = load_real_heterodata()

    print("Fitting per-table diffusion models + FK samplers...")
    artifacts = fit_all(real_data, schema, epochs=50, device=device)

    print("Generating a new synthetic graph...")
    synthetic_data = generate_graph(artifacts, schema, device=device)

    print(synthetic_data)
    torch.save(synthetic_data, "synthetic_claims_graph.pt")
    print("Saved to synthetic_claims_graph.pt")


if __name__ == "__main__":
    main()
