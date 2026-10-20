"""
Schema definition for the heterogeneous claims graph.

This module turns your PyG `HeteroData` schema (node_counts, feat_dims, relations)
into a config object the rest of the pipeline (diffusion.py, fk_sampler.py,
generate.py) consumes. It makes NO assumption about which columns inside each
node type's feature vector are continuous vs. categorical -- YOU must fill that
in once, based on your actual preprocessing, via `ColumnSpec`.

If you honestly don't know / didn't track it, a safe fallback is to treat every
column as continuous (see `default_all_continuous_schema` at the bottom) -- the
pipeline will still run, you'll just lose the benefit of multinomial diffusion
on the genuinely discrete columns (e.g. one-hot chronic-condition flags).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ColumnSpec:
    """Describes one node type's feature columns.

    continuous_idx: column indices (into the feat_dim-wide vector) that are
        real-valued (e.g. claim amount, age, deductible).
    categorical_idx: {column index -> number of classes} for columns that are
        discrete / class labels (e.g. a chronic-condition flag, a state code
        stored as an integer, a procedure-type bucket).

    `len(continuous_idx) + len(categorical_idx)` must equal the node type's
    feat_dim. Categorical columns are assumed to already be integer-encoded
    (0..num_classes-1) in your HeteroData `.x` tensor, not one-hot.
    """
    continuous_idx: List[int] = field(default_factory=list)
    categorical_idx: Dict[int, int] = field(default_factory=dict)

    def validate(self, feat_dim: int, name: str):
        cols = sorted(self.continuous_idx + list(self.categorical_idx.keys()))
        if cols != list(range(feat_dim)):
            raise ValueError(
                f"[{name}] continuous_idx + categorical_idx must partition "
                f"range(0, {feat_dim}) exactly. Got columns {cols}."
            )


@dataclass
class NodeTypeSpec:
    name: str
    num_nodes: int
    feat_dim: int
    columns: ColumnSpec

    def __post_init__(self):
        self.columns.validate(self.feat_dim, self.name)

    @property
    def n_continuous(self) -> int:
        return len(self.columns.continuous_idx)

    @property
    def n_categorical(self) -> int:
        return len(self.columns.categorical_idx)


@dataclass
class RelationSpec:
    src: str
    name: str
    dst: str
    # reverse relation name, e.g. 'rev_has_claim' for ('beneficiary','has_claim','claim')
    reverse_name: Optional[str] = None
    # is this a many-to-many "vocabulary attachment" relation (claim<->diagnosis,
    # claim<->procedure) as opposed to a simple many-to-one FK (claim->beneficiary)?
    many_to_many: bool = False


@dataclass
class GraphSchema:
    node_types: Dict[str, NodeTypeSpec]
    # only the *forward* relations the generator actually produces; reverse
    # edges are derived automatically (edge_index.flip(0))
    forward_relations: List[RelationSpec]

    def relation(self, src, name, dst) -> RelationSpec:
        for r in self.forward_relations:
            if (r.src, r.name, r.dst) == (src, name, dst):
                return r
        raise KeyError((src, name, dst))


# ---------------------------------------------------------------------------
# Your schema, wired up from the conversation.
# EDIT the ColumnSpec blocks below to match your real continuous/categorical
# column layout -- this is the one place domain knowledge has to be injected.
# ---------------------------------------------------------------------------

NODE_COUNTS = {
    "beneficiary": 138556,
    "claim": 558211,
    "provider": 5410,
    "physician": 100738,
    "diagnosis": 11015,
    "procedure": 1327,
}

FEAT_DIMS = {
    "beneficiary": 28,
    "claim": 4,
    "provider": 21,
    "physician": 3,
    "diagnosis": 3,
    "procedure": 3,
}

# Example split -- REPLACE with your real layout. As a placeholder, the last
# third of each vector is treated as categorical (binary/class columns),
# everything before that as continuous. This is almost certainly wrong for
# your actual preprocessing; it only exists so the code runs out of the box.
def _default_columns(feat_dim: int, n_cat: int, classes_per_cat: int = 2) -> ColumnSpec:
    n_cat = min(n_cat, feat_dim)
    cont = list(range(0, feat_dim - n_cat))
    cat = {i: classes_per_cat for i in range(feat_dim - n_cat, feat_dim)}
    return ColumnSpec(continuous_idx=cont, categorical_idx=cat)


def build_default_schema() -> GraphSchema:
    node_types = {
        "beneficiary": NodeTypeSpec(
            "beneficiary", NODE_COUNTS["beneficiary"], FEAT_DIMS["beneficiary"],
            _default_columns(28, n_cat=11, classes_per_cat=2),  # e.g. 11 chronic-condition flags
        ),
        "claim": NodeTypeSpec(
            "claim", NODE_COUNTS["claim"], FEAT_DIMS["claim"],
            _default_columns(4, n_cat=1, classes_per_cat=4),    # e.g. 1 claim-type class
        ),
        "provider": NodeTypeSpec(
            "provider", NODE_COUNTS["provider"], FEAT_DIMS["provider"],
            _default_columns(21, n_cat=5, classes_per_cat=2),
        ),
        "physician": NodeTypeSpec(
            "physician", NODE_COUNTS["physician"], FEAT_DIMS["physician"],
            _default_columns(3, n_cat=1, classes_per_cat=3),    # e.g. specialty class
        ),
        "diagnosis": NodeTypeSpec(
            "diagnosis", NODE_COUNTS["diagnosis"], FEAT_DIMS["diagnosis"],
            _default_columns(3, n_cat=3, classes_per_cat=2),    # diagnosis code often ~all categorical
        ),
        "procedure": NodeTypeSpec(
            "procedure", NODE_COUNTS["procedure"], FEAT_DIMS["procedure"],
            _default_columns(3, n_cat=3, classes_per_cat=2),
        ),
    }

    forward_relations = [
        RelationSpec("beneficiary", "has_claim", "claim", "rev_has_claim", many_to_many=False),
        RelationSpec("claim", "serviced_by", "provider", "rev_serviced_by", many_to_many=False),
        RelationSpec("claim", "attended_by", "physician", "rev_attended_by", many_to_many=False),
        RelationSpec("claim", "has_dx", "diagnosis", "rev_has_dx", many_to_many=True),
        RelationSpec("claim", "has_px", "procedure", "rev_has_px", many_to_many=True),
        RelationSpec("physician", "affiliated_with", "provider", "rev_affiliated_with", many_to_many=False),
        RelationSpec("beneficiary", "sees_physician", "physician", "rev_sees_physician", many_to_many=True),
    ]

    return GraphSchema(node_types=node_types, forward_relations=forward_relations)


# Topological generation order: independent "dimension" tables first, then
# tables that depend on already-generated parents. `claim` depends on
# beneficiary/provider/physician, which must therefore come first; diagnosis
# and procedure are independent vocabularies that can be generated any time
# before has_dx/has_px are sampled.
GENERATION_ORDER = ["provider", "physician", "diagnosis", "procedure", "beneficiary", "claim"]
