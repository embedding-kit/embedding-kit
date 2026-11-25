# Count Data & Alignment Losses

This page shows how to train on raw counts with likelihood-based losses, how to write a custom loss from the element-wise negative log-likelihoods, and how to align the embeddings of two paired data types.

**Precision.** Every loss on this page accepts `bfloat16` and `float16` tensors. The likelihoods are evaluated in float32 for those dtypes (lgamma and the log terms lose too much precision otherwise) and the result is cast back to the input dtype, so mixed-precision training works unchanged.

---

## Which loss for which data

The losses in the [Training Guide](../training.md) (`MSEVAELoss`, `BCEVAELoss`, `BCEWithLogitsVAELoss`) expect normalized inputs. When you want the model to see the data on its native scale, match the loss to how the data was generated:

| Data | Typical example | Use |
|------|-----------------|-----|
| Normalized, roughly Gaussian | log-normalized expression, CLR proteins, LSI | `MSEVAELoss` |
| Values in [0, 1] | min-max scaled expression | `BCEWithLogitsVAELoss` |
| Raw counts, little overdispersion | low-depth counts, spike-ins | `PoissonVAELoss` |
| Raw counts, overdispersed | single-cell UMI counts, bulk RNA-seq counts | `NBVAELoss` |
| Raw counts with excess zeros | very sparse counts | `zinb_nll` in a custom loss |
| Fractions in (0, 1) | methylation β-values, allele fractions | `beta_nll` in a custom loss |
| Successes out of trials | methylated reads / coverage | `binomial_nll`, `beta_binomial_nll` in a custom loss |

---

## Quick start: a negative binomial VAE

```python
import pandas as pd
import torch
from embkit import dataframe_loader, optimize
from embkit.models.vae import BaseVAE
from embkit.factory.layers import Layer
from embkit.losses import NBVAELoss

counts = pd.read_csv("counts.tsv", sep="\t", index_col=0)   # raw counts, rows = samples, columns = genes
loader = dataframe_loader(counts, batch_size=256)

vae = BaseVAE(
    features=list(counts.columns),
    latent_dim=32,
    encoder_layers=[Layer(256, activation="relu")],
    decoder_layers=[Layer(256, activation="relu")],   # no output activation: the loss maps outputs to NB means
)
loss_fn = NBVAELoss(n_features=counts.shape[1])        # one learned dispersion per gene

# the dispersion is a parameter of the loss, so give the optimizer both parameter sets
optimizer = torch.optim.Adam(list(vae.parameters()) + list(loss_fn.parameters()), lr=1e-3)
history = optimize.fit_vae(vae, X=loader, epochs=50, loss=loss_fn, optimizer=optimizer)

embedding = vae.encode(torch.tensor(counts.values, dtype=torch.float32))   # (samples, 32)
```

`fit_vae` works exactly as described in the [Training Guide](../training.md): beta schedules, gradient accumulation and the returned `history` are unchanged.

---

## NBVAELoss

```python
NBVAELoss(
    beta=1.0,               # KL weight
    n_features=None,        # set for one dispersion per feature; None = one shared dispersion
    theta=1.0,              # initial inverse dispersion
    learn_dispersion=True,  # False = keep theta fixed
)
```

The decoder output is turned into the NB mean with `softplus(recon)`, so the decoder must **not** end in a sigmoid or other bounded activation. The variance of each feature is `mean + mean² / theta`: small `theta` means strong overdispersion.

> **Train the dispersion.** With `learn_dispersion=True`, `theta` is stored as `loss_fn.log_theta`, a parameter of the **loss**, not the model. If you let `fit_vae` build its own optimizer, only the model is trained and `theta` stays at its initial value. Pass an optimizer over both, as in the quick start.

After training, the learned dispersions are:

```python
theta = torch.exp(loss_fn.log_theta.detach())   # (n_features,) or (1,)
```

They are not saved with the model by `factory.save`. Store them yourself if you need them later (for example `torch.save(loss_fn.state_dict(), "nb_loss.pt")`).

---

## PoissonVAELoss

```python
from embkit.losses import PoissonVAELoss

loss_fn = PoissonVAELoss(beta=1.0)
history = optimize.fit_vae(vae, X=loader, epochs=50, loss=loss_fn)
```

The decoder output is read as the **log-rate** of each feature. There are no extra parameters, so the default optimizer is fine.

---

## Writing a custom likelihood loss

The element-wise negative log-likelihoods return a tensor of the same shape as the input, so you choose how to reduce them:

| Function | Data | Parameters |
|----------|------|------------|
| `gaussian_nll(x, mean, logvar=None)` | continuous | mean, optional log-variance |
| `poisson_nll(x, log_rate)` | counts | log-rate |
| `nb_nll(x, mu, theta)` | counts | mean, inverse dispersion |
| `zinb_nll(x, mu, theta, logit_pi)` | counts with excess zeros | mean, inverse dispersion, zero-inflation logit |
| `beta_nll(x, alpha, beta)` | fractions in (0, 1) | two shape parameters |
| `binomial_nll(successes, total_count, logits)` | successes / trials | success logit |
| `beta_binomial_nll(successes, total_count, alpha, beta)` | overdispersed successes / trials | two shape parameters |

To use one with `fit_vae`, subclass `VAELoss` and return a `VAELossOutput`. This example models fractions (such as methylation β-values) with a Beta likelihood whose mean comes from the decoder and whose concentration is fixed:

```python
from embkit.losses import VAELoss, VAELossOutput, beta_nll

class BetaVAELoss(VAELoss):
    """Beta reconstruction for values in (0, 1); the decoder outputs the logit of the mean."""

    def __init__(self, beta: float = 1.0, concentration: float = 20.0):
        super().__init__(beta=beta)
        self.concentration = concentration

    def forward(self, recon, x, mu, logvar):
        mean = torch.sigmoid(recon)
        recon_loss = beta_nll(x, mean * self.concentration, (1 - mean) * self.concentration).sum(dim=1)
        kl_loss = self._kl_divergence(mu, logvar)
        total = (recon_loss + self.beta * kl_loss).mean()
        return VAELossOutput(total=total, recon=recon_loss.mean(), kl=kl_loss.mean())

fractions = pd.read_csv("methylation_beta.tsv", sep="\t", index_col=0)   # values in (0, 1)
meth_vae = BaseVAE(features=list(fractions.columns), latent_dim=16,
                   encoder_layers=[Layer(128, activation="relu")], decoder_layers=[Layer(128, activation="relu")])
optimize.fit_vae(meth_vae, X=dataframe_loader(fractions), epochs=50, loss=BetaVAELoss())
```

`fit_vae` passes only the input batch to the loss, so likelihoods that need a second matrix (successes **and** coverage) or several outputs per feature (ZINB) need your own training loop, as in the next section.

---

## Aligning two paired views

When the same samples are measured twice (RNA and protein from the same cells, two assays on the same tumours), train one VAE per view and add a term that pulls each sample's two embeddings together.

| Loss | Input | Use when |
|------|-------|----------|
| `pairwise_alignment_loss([mu_a, mu_b, ...])` | paired views | you want matching samples to share coordinates (mean squared distance; any number of views) |
| `info_nce_loss(mu_a, mu_b, temperature)` | paired views | you want each sample closer to its own partner than to every other sample in the batch (contrastive) |
| `mmd_loss(z_a, z_b)` | **unpaired** sets | two cohorts should have the same latent distribution, without sample-level pairs |
| `kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)` | two Gaussians | KL between two posteriors, or against a non-standard prior |

All except `mmd_loss` return one value per sample; call `.mean()` before `backward()`.

```python
from torch.utils.data import DataLoader, TensorDataset
from embkit.losses import MSEVAELoss, info_nce_loss

rna = pd.read_csv("rna_lognorm.tsv", sep="\t", index_col=0)
adt = pd.read_csv("adt_clr.tsv", sep="\t", index_col=0).loc[rna.index]   # same samples, same order

def make_vae(df):
    return BaseVAE(features=list(df.columns), latent_dim=16,
                   encoder_layers=[Layer(128, activation="relu")], decoder_layers=[Layer(128, activation="relu")])

rna_vae, adt_vae = make_vae(rna), make_vae(adt)
recon = MSEVAELoss(beta=0.5)
optimizer = torch.optim.Adam(list(rna_vae.parameters()) + list(adt_vae.parameters()), lr=1e-3)
pairs = TensorDataset(torch.tensor(rna.values, dtype=torch.float32), torch.tensor(adt.values, dtype=torch.float32))
loader = DataLoader(pairs, batch_size=128, shuffle=True, drop_last=True)   # InfoNCE needs >= 2 samples per batch

align_weight = 1.0
for epoch in range(50):
    for x_rna, x_adt in loader:
        out_rna, out_adt = rna_vae(x_rna), adt_vae(x_adt)
        loss = (recon(out_rna.recon, x_rna, out_rna.mu, out_rna.logvar).total
                + recon(out_adt.recon, x_adt, out_adt.mu, out_adt.logvar).total
                + align_weight * info_nce_loss(out_rna.mu, out_adt.mu, temperature=0.5).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Practical notes

- **Temperature.** Smaller values make InfoNCE focus on the hardest negatives; values between 0.1 and 1 are a reasonable range to try.
- **Batch size matters for InfoNCE.** Every other sample in the batch is a negative, so larger batches give a stronger signal. Use `drop_last=True` so a final batch of one sample does not raise an error.
- **Weighting.** Start with an alignment weight near 1 and check that reconstruction does not collapse; if one view dominates, lower the weight or warm up with reconstruction only.
- **Unpaired cohorts.** For `mmd_loss`, draw a batch from each cohort and pass their latent means. It is a set-level statistic, so use batches of at least a few dozen samples.

See the API pages for [likelihood losses](../api/losses/likelihood.md) and [alignment losses](../api/losses/alignment.md) for every argument.
