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

from graphgen.schema import build_schema_from_data
from graphgen.train import fit_all
from graphgen.generate import generate_graph

# Which column indices (into each node type's feature vector) are
# categorical. The NUMBER OF CLASSES for each is inferred automatically from
# your real data (max observed value + 1) by build_schema_from_data -- you
# only need to say which indices are categorical, not how many classes they
# have. Getting this wrong in the other direction (a continuous column
# listed as categorical) will also misbehave, so double check against your
# actual preprocessing/feature-engineering code.
CATEGORICAL_COLUMNS = {
    "beneficiary": [],   # e.g. [17,18,19,20,21,22,23,24,25,26,27]
    "claim": [],         # e.g. [3]
    "provider": [],      # e.g. [16,17,18,19,20]
    "physician": [],     # e.g. [2]
    "diagnosis": [],     # e.g. [0,1,2]
    "procedure": [],     # e.g. [0,1,2]
}


def load_real_heterodata() -> HeteroData:
    raise NotImplementedError(
        "Point this at your real HeteroData, e.g.:\n"
        "    return torch.load('claims_graph.pt')"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_data = load_real_heterodata()

    # EDIT CATEGORICAL_COLUMNS above to match your real feature layout before
    # running this for real -- class counts are inferred, but which columns
    # are categorical vs. continuous is domain knowledge only you have.
    schema = build_schema_from_data(real_data, categorical_columns=CATEGORICAL_COLUMNS)

    print("Fitting per-table diffusion models + FK samplers...")
    artifacts = fit_all(real_data, schema, epochs=50, device=device)

    print("Generating a new synthetic graph...")
    synthetic_data = generate_graph(artifacts, schema, device=device)

    print(synthetic_data)
    torch.save(synthetic_data, "synthetic_claims_graph.pt")
    print("Saved to synthetic_claims_graph.pt")


if __name__ == "__main__":
    main()
