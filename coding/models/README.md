# graphgen

Generates a new synthetic graph matching your claims `HeteroData` schema
(beneficiary / claim / provider / physician / diagnosis / procedure, mixed
continuous + categorical attributes), using a ClavaDDPM-style approach:
per-table mixed-type diffusion (TabDDPM-style) + cluster-guided foreign-key
assignment instead of dense adjacency-matrix diffusion. See the module
docstrings in `graphgen/diffusion.py` and `graphgen/fk_sampler.py` for why
this design was chosen over GDSS/DiGress-style graph diffusion at your scale
(claim=558K nodes).

## Install

```bash
pip install -r requirements.txt
```

## 1. Fill in your real column layout

Open `graphgen/schema.py` and edit the `_default_columns(...)` calls inside
`build_default_schema()` -- replace the placeholder continuous/categorical
split with your actual preprocessing (which columns in each `feat_dim`-wide
vector are real-valued vs. integer-coded classes, and how many classes each
categorical column has). This is the only piece of domain knowledge the code
can't infer on its own.

## 2. Fit

```python
import torch
from graphgen.schema import build_default_schema
from graphgen.train import fit_all

schema = build_default_schema()
real_data = torch.load("your_claims_graph.pt")  # your real HeteroData

artifacts = fit_all(real_data, schema, epochs=50, device="cuda")
```

This trains:
- one `MixedTypeDiffusion` per node type (Gaussian diffusion on continuous
  columns, multinomial/D3PM diffusion on categorical columns, shared MLP
  denoiser),
- one `ManyToOneFKSampler` per simple FK relation (beneficiary->claim,
  physician->provider): KMeans clusters parents by [attrs + child degree],
  an MLP classifier predicts cluster from attrs alone (for synthetic
  parents), and a per-cluster empirical degree distribution is recorded,
- one `VocabAttachmentSampler` per many-to-many / shared-vocabulary relation
  (claim<->diagnosis, claim<->procedure, claim->provider, claim->physician,
  beneficiary<->physician): per-cluster empirical frequency over vocabulary
  ids + per-cluster cardinality distribution.

## 3. Generate

```python
from graphgen.generate import generate_graph

synthetic = generate_graph(artifacts, schema, device="cuda")
torch.save(synthetic, "synthetic_claims_graph.pt")
```

`generate_graph` builds the new `HeteroData` by: sampling independent
dimension tables unconditionally -> sampling beneficiary clusters/degrees ->
generating claim rows conditioned on beneficiary cluster -> sampling
claim->provider / claim->physician / claim<->diagnosis / claim<->procedure /
physician->provider / beneficiary<->physician edges directly as index
arrays from the fitted samplers. No N×N adjacency matrix is ever formed, so
this scales to your real node counts.

## 4. Validate (do this before trusting the output)

At minimum, check on a held-out split of the real data:
- per-column marginal distributions (KS-test or histogram overlay) for
  continuous columns, chi-square for categorical columns, real vs. synthetic
- degree distributions per relation, real vs. synthetic (claims-per-
  beneficiary, diagnoses-per-claim, etc.)
- a simple "train on synthetic, test on real" check for whatever downstream
  task you ultimately care about (e.g. a fraud/claim-outcome classifier) --
  this is the real test of whether the synthetic graph is useful, not just
  whether marginals look similar.

## Notes / things to tune

- `n_clusters` (default 16) controls how much parent/child correlation is
  preserved vs. how much data each cluster's sub-model gets to train on --
  raise it for a larger dataset, lower it if clusters end up with too few
  real examples.
- `T` (diffusion steps, default 1000) and `epochs` in `train_node_diffusion`
  trade training time for sample fidelity; start smaller (`T=200`,
  `epochs=20`) to get the pipeline working end-to-end before scaling up.
- The claim->provider and claim->physician assignment currently goes through
  `VocabAttachmentSampler` with cardinality naturally ~1 (since each real
  claim has one provider/physician). If your data has claims with multiple
  providers, that's handled automatically by the learned cardinality
  distribution -- no code change needed.
