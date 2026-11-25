# Spatial Data & Analysis Utilities

This page shows how to prepare spatial assays (Visium, Xenium, CosMx, imaging mass cytometry) for embedding models, how to match samples between two datasets, how to attribute a trained model's output to its input features, and how to make runs reproducible.

**Precision.** Coordinates, features, images and embeddings can be NumPy arrays or torch tensors of any dtype, including `bfloat16`; the array outputs are NumPy. `integrated_gradients` calls your model with tensors of the input's dtype and returns attributions in that dtype.

---

## Spatial data at a glance

Spatial assays attach a position to every spot or cell. The functions here work on plain arrays — a `(spots, features)` matrix and a `(spots, 2)` coordinate array — so they fit any reader.

| Goal | Function |
|------|----------|
| give the model each spot's surroundings | `neighborhood_aggregate` |
| use the tissue image | `extract_tiles` |
| keep neighbouring spots in the same mini-batch | `SpatialBlockSampler` |
| inspect the neighbour graph itself | `knn_graph`, `neighborhood_weights`, `median_neighbor_distance` |

---

## Niche features

`neighborhood_aggregate` averages each spot's spatial neighbours into a second "niche" matrix of the same shape. Concatenating it with the spot's own profile lets an ordinary VAE see both:

```python
import numpy as np
import pandas as pd
from embkit.spatial import neighborhood_aggregate, median_neighbor_distance

expr = pd.read_csv("spots_lognorm.tsv", sep="\t", index_col=0)       # (spots, genes)
meta = pd.read_csv("spots_meta.tsv", sep="\t", index_col=0).loc[expr.index]
coords = meta[["x", "y"]].to_numpy()

niche = neighborhood_aggregate(expr.values, coords, k=6, groups=meta["slide"].values)
niche_df = pd.DataFrame(niche, index=expr.index, columns=[f"niche_{g}" for g in expr.columns])
model_input = pd.concat([expr, niche_df], axis=1)                    # train any embkit VAE on this
```

| Argument | Default | Notes |
|----------|---------|-------|
| `k` | 6 | neighbours per spot; 6 matches the hexagonal Visium grid |
| `kernel` | `"uniform"` | `"gaussian"` weights closer neighbours more, with width `sigma` (default: the median nearest-neighbour distance) |
| `include_self` | `False` | add the spot itself to its neighbourhood |
| `groups` | `None` | one label per spot (e.g. slide); neighbours are searched within each group, so slides placed on overlapping coordinates do not mix |

Dense input gives a dense result and sparse input a sparse one. `median_neighbor_distance(coords)` tells you the typical spot spacing in your coordinate units.

---

## Image tiles

`extract_tiles` cuts a square tile around every spot from the tissue image and returns them flat, in the channel-first `C × H × W` layout that `embkit.modules.conv.ConvBackbone` expects:

```python
from PIL import Image
from embkit.spatial import extract_tiles, unflatten_tiles

image = np.asarray(Image.open("tissue_image.png").convert("RGB"))   # (H, W, 3), uint8
centers = meta[["pixel_x", "pixel_y"]].to_numpy()                   # (x, y) = (column, row) in THIS image's pixels

tiles = extract_tiles(image, centers, tile_size=64, out_size=32)    # (spots, 3 * 32 * 32), values in [0, 1]
tiles_4d = unflatten_tiles(tiles, (3, 32, 32))                      # (spots, 3, 32, 32), for plotting
```

- **Coordinate scale.** Spot coordinates are often given in full-resolution pixels while the image you load is downscaled (for Visium, multiply by the scale factor of the image you use). Centres outside the image raise an error, which usually means the scale is wrong.
- **Borders.** Tiles that run past the edge are completed by reflecting the image.
- **Resizing.** `out_size` averages pixel blocks when it divides `tile_size`, and uses antialiased bilinear resizing otherwise.
- Integer images are scaled to [0, 1]; float images are used as they are.

For stain differences between slides, run the tiles through `MacenkoNormalizer` from `embkit.preprocessing` one slide at a time before training.

---

## Spatially local mini-batches

Spatial smoothness terms compare neighbouring spots, so they need neighbours in the same mini-batch. `SpatialBlockSampler` builds each batch from a random seed spot plus its nearest spots:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from embkit.spatial import SpatialBlockSampler

dataset = TensorDataset(torch.tensor(model_input.values, dtype=torch.float32))
sampler = SpatialBlockSampler(coords, batch_size=256, groups=meta["slide"].values, seed=0)
loader = DataLoader(dataset, batch_sampler=sampler)   # batch_sampler=, not sampler=

for (x,) in loader:
    pass                                              # every batch is one spatial neighbourhood
```

Batches never span groups. Every spot is visited each epoch (groups with a single spot are skipped); because neighbourhoods overlap, some spots appear in more than one batch. Each epoch uses a new order (`seed + epoch`).

---

## Matching samples between datasets

When two datasets are not paired sample-by-sample, project both into a shared space first — a PCA fitted on shared features, CCA, or a jointly trained latent — then find pseudo-pairs:

```python
from sklearn.decomposition import PCA
from embkit.matching import bipartite_match, mnn_anchors

a = pd.read_csv("cohort_a_lognorm.tsv", sep="\t", index_col=0)
b = pd.read_csv("cohort_b_lognorm.tsv", sep="\t", index_col=0)
shared = a.columns.intersection(b.columns)
pca = PCA(n_components=20).fit(pd.concat([a[shared], b[shared]]))
XA, XB = pca.transform(a[shared]), pca.transform(b[shared])

idx_a, idx_b = bipartite_match(XA, XB)        # one-to-one, minimizing total distance
anc_a, anc_b = mnn_anchors(XA, XB, k=20)      # mutual nearest neighbours: fewer, higher-confidence pairs
pairs = pd.DataFrame({"a": a.index[anc_a], "b": b.index[anc_b]})
```

| Function | Gives | Use when |
|----------|-------|----------|
| `bipartite_match` | a full one-to-one assignment | you need a partner for (nearly) every sample, for example to train a paired model |
| `mnn_anchors` | only pairs that are each in the other's k nearest neighbours | you want conservative anchors and can accept leaving many samples unmatched |

`bipartite_match` uses the Hungarian algorithm, which scales as `n³`, so both sides are subsampled to `max_samples` (default 20,000) first. Returned indices are positions in `XA` / `XB`.

---

## Which features drive the model: integrated gradients

`integrated_gradients` attributes any scalar (or per-sample) output of a model to its input features. For a trained VAE, explain one latent dimension:

```python
from embkit.interpret import integrated_gradients
from embkit.models.vae import BaseVAE
from embkit.factory.layers import Layer

vae = BaseVAE(features=list(expr.columns), latent_dim=8,
              encoder_layers=[Layer(64, activation="relu")], decoder_layers=[Layer(64, activation="relu")])
vae.eval()   # use your trained model; eval mode makes the encoder deterministic

x = torch.tensor(expr.values[:100], dtype=torch.float32)
attr = integrated_gradients(lambda inp: vae.encoder(inp).mu[:, 3], x, steps=32)   # latent dimension 3
top_genes = pd.Series(attr.abs().mean(0).numpy(), index=expr.columns).nlargest(20)
```

Attributions for each sample sum (approximately) to the output at that sample minus the output at the baseline. The default baseline is all zeros; pass `baseline=` for a more meaningful reference, such as the mean profile of a control group. Increase `steps` if the sums are far from that difference.

---

## Reproducible runs

```python
from embkit.utilities import set_seed

set_seed(0)                       # Python, NumPy and PyTorch (CPU and all GPUs)
set_seed(0, deterministic=True)   # also force deterministic cuDNN kernels (slower)
```

See the API pages for [spatial utilities](../api/analysis/spatial.md), [matching](../api/analysis/matching.md) and [integrated gradients](../api/analysis/interpret.md).
