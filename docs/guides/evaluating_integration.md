# Evaluating an Integration

This page shows how to score an embedding that integrates two data types measured on the same samples, and how to check reconstruction quality feature by feature.

**Precision.** Every metric on this page accepts NumPy arrays or torch tensors of any dtype, including `bfloat16` and integer counts. They compute in float64 and return NumPy values.

---

## What you need

Embed held-out samples **once per data type** into the shared latent space, so that row `i` of each matrix is the same sample:

```python
import numpy as np

Z_rna = np.load("z_rna_test.npy")   # (n_samples, latent_dim), from the RNA encoder
Z_adt = np.load("z_adt_test.npy")   # (n_samples, latent_dim), from the protein encoder
```

Use the per-view embeddings, not a fused one. A fused embedding gives both views of a sample the same point, which makes every alignment score perfect and meaningless.

---

## Quick start

```python
from embkit.metrics.alignment import foscttm, match_recall_at_k, modality_mixing, modality_entropy

fos = foscttm(Z_rna, Z_adt)                                   # lower is better
recall = {k: match_recall_at_k(Z_rna, Z_adt, k=k) for k in (1, 10, 50)}

Z_all = np.vstack([Z_rna, Z_adt])
view = np.repeat(["rna", "adt"], len(Z_rna))
mixing = modality_mixing(Z_all, view, k=20)
entropy = modality_entropy(Z_all, view, k=30)
```

---

## What the metrics mean

| Metric | Function | Measures | Better | Caveat |
|--------|----------|----------|--------|--------|
| FOSCTTM | `foscttm` | for each sample, the fraction of samples closer to it than its true partner | lower: 0 is perfect, about 0.5 is random | needs paired samples |
| Recall@k | `match_recall_at_k` | fraction of samples whose true partner is among their k nearest cross-view neighbours | higher | depends on k; report several |
| Modality mixing | `modality_mixing` | share of each sample's neighbours (in the stacked embedding) that come from the other view | higher, up to the balanced value | perfect mixing can come from a collapsed latent |
| Modality entropy | `modality_entropy` | entropy of view labels among neighbours, scaled to [0, 1] | higher | same caveat as mixing |

Both FOSCTTM and recall@k compare one view against the other, so they are not symmetric. Compute both directions and report the worse one if they differ:

```python
fos_rna_to_adt = foscttm(Z_rna, Z_adt)
fos_adt_to_rna = foscttm(Z_adt, Z_rna)
```

### Error bars and per-sample values

Every function returns the mean by default. Ask for the standard error and/or the per-sample values to plot distributions or find badly aligned samples:

```python
mean, sem = foscttm(Z_rna, Z_adt, return_sem=True)
mean, sem, per_sample = foscttm(Z_rna, Z_adt, return_sem=True, return_per_sample=True)
worst = np.argsort(per_sample)[-20:]   # the 20 worst-aligned samples
```

`metric="cosine"` is available for `foscttm` and `match_recall_at_k`; the mixing functions accept any scikit-learn metric name.

---

## Reconstruction quality per feature

A single reconstruction loss hides which features the model reproduces and which it flattens. Score each feature:

```python
import pandas as pd
from embkit.metrics.reconstruction import reconstruction_metrics

x_true = pd.read_csv("adt_test.tsv", sep="\t", index_col=0)
x_pred = pd.read_csv("adt_test_predicted.tsv", sep="\t", index_col=0)   # e.g. protein predicted from RNA

rec = reconstruction_metrics(x_true.values, x_pred.values)
rec["pearson_median"], rec["mse_median"]
per_protein = pd.Series(rec["pearson_per_feature"], index=x_true.columns).sort_values()
```

| `kind` | Returns |
|--------|---------|
| `"continuous"` (default) | `mse_mean`, `mse_median`, `pearson_mean`, `pearson_median`, plus `mse_per_feature` and `pearson_per_feature` arrays |
| `"binary"` | `auc_mean`, `auc_median`, `auc_per_feature` (for 0/1 data such as binarized peaks; constant features get NaN) |

The per-feature pieces are also available on their own: `mse_per_feature`, `pearson_corr_per_feature` and `auc_per_feature_binary`.

### Flagging samples the model cannot explain

`deviation_scores` turns per-sample losses into robust z-scores against a reference population, for example reconstruction errors of held-out healthy samples:

```python
from embkit.metrics.reconstruction import deviation_scores

err_healthy = ((x_true.values[:100] - x_pred.values[:100]) ** 2).mean(axis=1)
err_query = ((x_true.values[100:] - x_pred.values[100:]) ** 2).mean(axis=1)
z = deviation_scores(err_query, err_healthy)   # (x - median) / (1.4826 * MAD)
outliers = x_true.index[100:][z > 3]
```

---

## Practical notes

- **Use held-out samples** for every metric. Training samples overstate alignment.
- **Subsample large test sets.** FOSCTTM and recall@k compare every sample against every other one. Memory is bounded (`block_size` rows at a time), but run time grows with `n²`; 10,000–20,000 paired samples give stable estimates.
- **Read mixing next to a biology metric.** High mixing alone does not show that biology is preserved. Pair it with label transfer or label purity from `embkit.metrics.neighbors`.
- **Report per-feature correlations** for cross-modal prediction, not only the average, and check that known markers are among the well-predicted features.

See the API pages for [integration metrics](../api/metrics/alignment.md) and [reconstruction metrics](../api/metrics/reconstruction.md).
