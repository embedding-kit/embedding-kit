# Preparing Assay Data

This page shows how to turn raw count matrices from RNA, surface-protein and chromatin-accessibility assays into model inputs, how to split samples so every class is represented, and how to normalize H&E stain colour across slides. Everything works on plain pandas DataFrames (or NumPy arrays); no AnnData or Scanpy is needed.

---

## Which transform for which assay

| Assay | Input | Transform | Output |
|-------|-------|-----------|--------|
| RNA (scRNA-seq, bulk) | raw counts | `library_size_norm` | counts per 10k, then `log1p` |
| Surface protein (CITE-seq ADT) | raw antibody counts | `clr_norm` | per-sample centred log-ratio |
| Chromatin accessibility (ATAC) | raw peak or bin counts | `TfidfLSI` | TF-IDF, then truncated SVD (LSI) components |
| H&E image tiles | RGB tiles in [0, 1] | `MacenkoNormalizer` | tiles re-rendered with common stain colours |

The outputs are log-scale or centred and can be negative, so train on them with `MSEVAELoss`. If you want BCE losses, scale to [0, 1] first with `MinMaxScaler` or `ExpMinMaxScaler` from `embkit.preprocessing`. To model raw counts directly instead, skip these transforms and use the count losses in `embkit.losses` (`NBVAELoss`, `PoissonVAELoss`).

---

## Split first, then fit

Anything fitted to the data — TF-IDF weights, LSI components, scalers — must be fitted on training samples only and then applied to validation, test and query samples. Otherwise information leaks from the test set into the model. Split before you fit:

```python
import numpy as np
import pandas as pd
from embkit.preprocessing import split_by_label

labels = pd.read_csv("labels.tsv", sep="\t", index_col=0)["cell_type"]
split = split_by_label(labels.values, train_fraction=0.8, val_fraction=0.1, seed=0)
train_ids = labels.index[split["train"]]
val_ids = labels.index[split["val"]]
test_ids = labels.index[split["test"]]
```

`split_by_label` shuffles **within** each label, so rare classes appear in every split in proportion. The split is deterministic for a given `seed`.

| Argument | Effect |
|----------|--------|
| `train_fraction`, `val_fraction` | fractions of each label's samples; the rest go to test (`train_fraction + val_fraction` must be < 1) |
| `train_cap`, `val_cap` | at most this many training / validation samples per label; the overflow goes to test |
| `max_per_label` | cap each label's pool before the fractions are applied, so large classes do not dominate (cannot be combined with the per-split caps) |

Missing labels raise an error; fill or drop them explicitly before splitting. Caps must be non-negative integers (or `None`), so a typo such as `train_cap=-1` raises instead of silently overlapping the splits.

---

## RNA: library-size normalization

```python
from embkit.preprocessing import library_size_norm

rna_counts = pd.read_csv("rna_counts.tsv", sep="\t", index_col=0)   # rows = samples, columns = genes
rna = library_size_norm(rna_counts, target_sum=1e4)                 # same index and columns
```

Each sample is scaled to `target_sum` total counts, removing sequencing-depth differences, then `log1p` is applied (`log1p=False` skips it). It is a per-sample operation with nothing to fit, so it can run before or after splitting. Select genes (for example the most variable ones in the training samples) before or after normalizing, as you prefer.

---

## Surface protein: CLR

```python
from embkit.preprocessing import clr_norm

adt_counts = pd.read_csv("adt_counts.tsv", sep="\t", index_col=0)
adt = clr_norm(adt_counts)
```

Each sample's `log1p` counts are centred on their own mean, so a cell that stained more brightly overall keeps the same profile. Like library-size normalization, it is per-sample. Pass raw counts, not values that are already CLR-transformed.

---

## Chromatin accessibility: TF-IDF + LSI

`TfidfLSI` follows the scikit-learn `fit` / `transform` pattern:

```python
from embkit.preprocessing import TfidfLSI

atac_counts = pd.read_csv("atac_counts.tsv", sep="\t", index_col=0)   # peaks or genome bins as columns

lsi = TfidfLSI(n_components=50, method="signac", drop_first=True, min_fraction=0.01)
atac_train = lsi.fit_transform(atac_counts.loc[train_ids])   # DataFrame with LSI_1 ... LSI_49
atac_test = lsi.transform(atac_counts.loc[test_ids])

kept_peaks = atac_counts.columns[lsi.features_]               # peaks that passed the detection filter
```

| Argument | Default | Notes |
|----------|---------|-------|
| `n_components` | 100 | SVD components fitted **before** any are dropped; must be smaller than the number of training samples and retained peaks |
| `method` | `"sklearn"` | `"sklearn"` (scikit-learn's TF-IDF), `"signac"` (`log1p(TF × IDF × 1e4)`), or `"tea"` (L1 TF × log IDF) |
| `drop_first` | `False` | the first LSI component usually tracks sequencing depth; dropping it is common practice |
| `scale` | `True` | standardize each component using training-set statistics |
| `min_fraction`, `max_fraction` | 0.0, 1.0 | keep peaks detected in this range of training samples |

Query samples must be quantified on the **same peak set** as the training samples; independently called peaks are not comparable. When you fit on a DataFrame, `transform` matches columns by name: a query DataFrame with the same peaks in a different order is put back in training order, and missing or extra peaks raise an error that names a few of them. A NumPy array has no names, so its columns must already be in training order.

---

## H&E stain normalization

Slides stained in different batches differ in colour for reasons unrelated to the tissue, and an image encoder will learn those differences. `MacenkoNormalizer` estimates each slide's haematoxylin and eosin colours and re-renders tiles with one common pair:

```python
from embkit.preprocessing import MacenkoNormalizer

reference_tiles = np.load("slide_A_tiles.npy")      # (n, 3, H, W), RGB in [0, 1]
other_tiles = np.load("slide_B_tiles.npy")

normalizer = MacenkoNormalizer().fit(reference_tiles)   # target appearance
slide_b = normalizer.transform(other_tiles)             # same shape, float32
```

- **One slide at a time.** Stain colours are estimated from the pixels passed in, so call `transform` once per slide (or staining batch), not on a mix of slides.
- **Background is ignored.** Pixels whose optical density is below `od_threshold` (default 0.15) in every channel are treated as glass. A call on tiles that are almost all background raises an error.
- **Large slides.** The estimate uses at most `max_pixels` (default 200,000) randomly chosen tissue pixels, so it stays fast.

Tiles must be finite: a NaN or infinite pixel raises an error naming the problem. Tiles can be NumPy arrays or torch tensors, including `bfloat16`; the output is always a float32 NumPy array.

The method is from Macenko et al., "A method for normalizing histology slides for quantitative analysis", IEEE ISBI 2009, pp. 1107–1110 ([doi:10.1109/ISBI.2009.5193250](https://doi.org/10.1109/ISBI.2009.5193250)).

`rgb_to_od` and `od_to_rgb` convert between intensity and optical density if you need them directly.

---

## Putting it together

```python
from embkit import dataframe_loader, optimize
from embkit.models.vae import BaseVAE
from embkit.factory.layers import Layer
from embkit.losses import MSEVAELoss

rna_train = rna.loc[train_ids]
vae = BaseVAE(features=list(rna_train.columns), latent_dim=32,
              encoder_layers=[Layer(256, activation="relu")], decoder_layers=[Layer(256, activation="relu")])
optimize.fit_vae(vae, X=dataframe_loader(rna_train), epochs=50, loss=MSEVAELoss())
```

See the [preprocessing API page](../api/preprocessing.md) for every argument.
