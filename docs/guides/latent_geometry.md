# Hyperbolic Latent Spaces & Hierarchy

This page covers training with a hyperbolic or product latent space, measuring distances in it, scoring an embedding against a known label tree, recovering hierarchy from an embedding, and the neighbourhood metrics (label transfer, label purity, batch mixing) that work in any of these geometries.

**Precision.** The distance, radius, hierarchy and neighbour functions accept NumPy arrays or torch tensors of any float dtype, including `bfloat16`. They compute in float64 and return NumPy arrays.

---

## When to use a hyperbolic latent space

A Euclidean latent space has room that grows polynomially with radius; a hyperbolic one has room that grows exponentially. Tree-like data — cell lineages, tumour subclones, a label taxonomy — fits into hyperbolic space with much less distortion, with general states near the origin and specialised ones further out.

| `geometry` | Latent columns | Use when |
|------------|----------------|----------|
| `"euclidean"` | all Euclidean | the default; no strong hierarchy |
| `"hyperbolic"` | all hyperbolic | the main structure is a tree |
| `"product"` | first block Euclidean, last `hyperbolic_dim` columns hyperbolic | a hierarchy plus continuous variation (cell cycle, depth, batch) |

```python
from embkit.geometry import split_latent_dims

split_latent_dims(16, "product")                     # (8, 8): half hyperbolic by default
split_latent_dims(16, "product", hyperbolic_dim=4)   # (12, 4)
```

### Coordinate convention

Latent vectors stay plain `(samples, latent_dim)` arrays. The hyperbolic block is stored as **tangent coordinates at the origin**, so:

- its norm is the geodesic distance from the origin (`latent_radius`);
- Euclidean distance between two hyperbolic blocks is **not** the geodesic distance — use `latent_distance` / `pairwise_latent_distance`;
- `curvature` (default 1.0) sets the curvature of the hyperbolic block to `-curvature`.

---

## Training a hyperbolic VAE

`fit_vae` samples a Gaussian latent, so a hyperbolic latent needs a short custom training loop. Reuse an embkit encoder and decoder, and replace the sampling and KL with their wrapped-normal versions:

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from embkit.models.vae import BaseVAE
from embkit.factory.layers import Layer
from embkit.geometry import wrapped_normal_rsample, wrapped_normal_kl

df = pd.read_csv("expression_norm.tsv", sep="\t", index_col=0)
x_all = torch.tensor(df.values, dtype=torch.float32)

vae = BaseVAE(features=list(df.columns), latent_dim=8,
              encoder_layers=[Layer(128, activation="relu")], decoder_layers=[Layer(128, activation="relu")])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(x_all), batch_size=128, shuffle=True)
curvature, beta = 1.0, 0.5

for epoch in range(50):
    for (x,) in loader:
        enc = vae.encoder(x)
        t, _ = wrapped_normal_rsample(enc.mu, enc.logvar, curvature)   # sample on the hyperboloid
        recon = vae.decoder(t)                                          # decoder reads tangent coordinates
        recon_loss = ((recon - x) ** 2).sum(dim=1)
        kl = wrapped_normal_kl(enc.mu, enc.logvar, curvature=curvature)  # vs. standard wrapped normal at the origin
        loss = (recon_loss + beta * kl).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

vae.eval()
z = vae.encode(x_all).numpy()   # posterior means, in tangent coordinates
```

For a **product** space, split `enc.mu` / `enc.logvar` with `split_latent_dims`, sample the Euclidean block with the usual reparameterisation and its closed-form KL, sample the hyperbolic block as above, and concatenate the two blocks (Euclidean first) before decoding.

`wrapped_normal_kl` has no closed form, so it is a one-sample Monte Carlo estimate: unbiased but noisy, and occasionally slightly negative for single rows. That is expected; it averages out over a batch.

---

## Distances, centroids and plotting

```python
from embkit.geometry import latent_distance, pairwise_latent_distance, latent_radius, latent_centroid, to_poincare

D = pairwise_latent_distance(z, geometry="hyperbolic")            # (n, n) geodesic distances
d = latent_distance(z[:10], z[10:20], geometry="hyperbolic")      # row-wise, (10,)
r = latent_radius(z)                                              # distance from the origin
c = latent_centroid(z, geometry="hyperbolic")                     # stays on the manifold
```

`pairwise_latent_distance` works through the rows in chunks (`chunk=1024` by default), so memory grows with `n × chunk`, not `n²`, while it computes — but the returned matrix is still `n × n`. For tens of thousands of samples, work on a subsample or use the neighbourhood functions below, which never build the full matrix.

To plot a 2-D hyperbolic block, map it into the Poincaré disk:

```python
import matplotlib.pyplot as plt

p = to_poincare(z[:, :2], geometry="hyperbolic")   # points inside the unit disk
plt.scatter(p[:, 0], p[:, 1], s=4)
plt.gca().add_patch(plt.Circle((0, 0), 1, fill=False))
plt.gca().set_aspect("equal")
```

---

## Scoring an embedding against a label tree

Describe the known hierarchy as `{node: parent}`, with `None` for the root:

```python
import numpy as np
from embkit.geometry import tree_distance_matrix, hierarchy_metrics

parents = {
    "HSC": None,
    "MPP": "HSC",
    "CMP": "MPP", "CLP": "MPP",
    "Mono": "CMP", "Ery": "CMP",
    "B": "CLP", "T": "CLP",
}
dist, names, depth = tree_distance_matrix(parents)        # optional lengths={node: edge length}

cell_types = pd.read_csv("labels.tsv", sep="\t", index_col=0)["cell_type"].loc[df.index]
codes = np.array([names.index(c) if c in names else -1 for c in cell_types])   # -1 = not in the tree
scores = hierarchy_metrics(z, codes, dist, depth, geometry="hyperbolic")
```

| Score | Meaning | Better |
|-------|---------|--------|
| `dist_spearman` | rank agreement between latent distances and tree distances over random sample pairs | higher (1 = same ordering) |
| `rel_distortion` | mean relative error after the best single rescaling of the tree | lower |
| `radius_depth` | rank agreement between latent radius and tree depth (only with `depth`) | higher: general states near the origin |

Run the same call with `geometry="euclidean"` on a Euclidean model to compare the two spaces on equal terms.

---

## Recovering hierarchy from an embedding

```python
from embkit.hierarchy import cluster_tree, tree_agreement, graph_pseudotime

tree = cluster_tree(z, cell_types.values, root="HSC", geometry="hyperbolic")
tree["parents"]        # {label: parent label or None}
tree["edge_length"]    # geodesic distance from each label's centroid to its parent's

agreement = tree_agreement(tree["parents"], parents)
# {"edge_jaccard": ..., "path_spearman": ..., "parent_accuracy": ...}

root_mask = (cell_types.values == "HSC")
pseudotime = graph_pseudotime(z, root_mask, k=15, geometry="hyperbolic")
```

- `cluster_tree` is a minimum spanning tree over label centroids. It always returns a tree and joins each cluster to its nearest neighbour, so it can link two close siblings to each other. Judge it against a reference with `tree_agreement` rather than reading it as ground truth.
- `tree_agreement` needs both trees to contain exactly the same nodes. `edge_jaccard` ignores the root; `parent_accuracy` depends on it.
- `graph_pseudotime` is the shortest-path distance from the root samples along a kNN graph. Disconnected parts of the graph are joined by their shortest connecting edge, so every sample gets a finite value.

---

## Neighbourhood metrics in any geometry

These live in `embkit.metrics.neighbors` and accept the same `geometry`, `hyperbolic_dim` and `curvature` arguments:

```python
from embkit.metrics.neighbors import transfer_labels, label_purity, batch_mixing, label_transfer_knn

purity = label_purity(z, cell_types.values, k=15, geometry="hyperbolic")
batches = pd.read_csv("labels.tsv", sep="\t", index_col=0)["batch"].loc[df.index].values
mixing = batch_mixing(z, batches, k=15, geometry="hyperbolic")

ref, query = slice(0, 200), slice(200, None)
calls = transfer_labels(z[ref], cell_types.values[ref], z[query], k=15, geometry="hyperbolic")
calls.head()           # columns: label, confidence, novelty
```

| Function | Returns | Read it as |
|----------|---------|------------|
| `label_purity` | mean share of each sample's neighbours with its own label | biology kept together (higher is better) |
| `batch_mixing` | batch mixing relative to what batch sizes allow: 0 = separate, ~1 = fully mixed | technical groups mixed (higher is better) |
| `transfer_labels` | per query sample: majority label, `confidence`, `novelty` | `novelty` well above 1 = the reference does not cover that sample, whatever its confidence |
| `label_transfer_knn` | predictions, accuracy, confusion matrix (optional macro/weighted F1) | Euclidean scikit-learn metrics, for comparison with other tools |

> **Read purity and mixing together.** A collapsed latent space mixes batches perfectly and is useless; high `batch_mixing` is only good news alongside high `label_purity`.

See the API pages for [latent geometry](../api/geometry.md), [hierarchy](../api/analysis/hierarchy.md) and [neighbourhood metrics](../api/metrics/neighbors.md).
