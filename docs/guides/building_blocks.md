# Model Building Blocks

This page shows how to build convolutional VAEs for image tiles and 1-D signals, fuse the posteriors of several encoders, remove a nuisance signal adversarially, and segment samples along an ordering with a hidden Markov model.

**Precision.** The modules on this page are ordinary `nn.Module`s and run in whatever dtype you move them to. `segment_by_order` accepts NumPy arrays or torch tensors of any float dtype, including `bfloat16`, and fits the HMM in float64.

---

## Convolutional VAEs

`ConvBackbone` and `ConvDecoder` let a VAE use convolutions while the data stays a flat `(samples, features)` matrix in channel-first order: `features = C × H × W` for images, `C × L` for 1-D signals. Loaders, losses and `factory.save` / `factory.load` therefore work unchanged.

### Quick start: image tiles

```python
import numpy as np
import torch
from embkit import optimize
from embkit.models.vae import VAE, VAEEncoder
from embkit.modules.conv import ConvBackbone, ConvDecoder
from embkit.losses import BCEWithLogitsVAELoss

tiles = np.load("tiles.npy")                        # (n_tiles, 3 * 32 * 32), values in [0, 1]
X = torch.tensor(tiles, dtype=torch.float32)

backbone = ConvBackbone(in_channels=3, spatial_shape=(32, 32), channels=(32, 64, 128),
                        hidden_dim=256, augment=("flip", "rot90"))
encoder = VAEEncoder(backbone, feature_dim=backbone.feature_dim, latent_dim=16)
decoder = ConvDecoder(latent_dim=16, in_channels=3, spatial_shape=(32, 32), channels=(32, 64, 128))
model = VAE(encoder=encoder, decoder=decoder)

history = optimize.fit_vae(model, X=X, epochs=30, loss=BCEWithLogitsVAELoss(), batch_size=128)
model.eval()
embedding = model.encode(X)                         # (n_tiles, 16)
```

The decoder outputs logits here (`output_activation="none"`, the default), which is what `BCEWithLogitsVAELoss` expects. Use `output_activation="sigmoid"` with `BCEVAELoss`, or keep `"none"` with `MSEVAELoss` for standardized intensities.

### Choosing the settings

| Argument | Default | Notes |
|----------|---------|-------|
| `channels` | `(32, 64, 128)` | one entry per stage; each stage halves the resolution, so every spatial size must be divisible by `2 ** len(channels)` |
| `pool` | `"flatten"` | `"mean"` averages over positions: translation-invariant and much smaller |
| `hidden_dim` | `256` | dense layer after pooling; `None` returns the pooled features directly |
| `norm` | `"group"` | `"batch"` or `"none"`; group norm is safest for small batches |
| `augment` | `()` | `"flip"` and/or `"rot90"`, training only, 2-D only; use when tile orientation carries no meaning (most microscopy) |
| `upsample` (decoder) | `"nearest"` | `"transpose"` uses transposed convolutions instead |

### 1-D signals

Give a one-element `spatial_shape` for ordered signals such as coverage tracks or spectra:

```python
signal_backbone = ConvBackbone(in_channels=2, spatial_shape=(1024,), channels=(16, 32, 64), pool="mean")
signal_decoder = ConvDecoder(latent_dim=8, in_channels=2, spatial_shape=(1024,), channels=(16, 32, 64))
```

### Saving and loading

```python
from embkit.factory import save, load
import embkit.modules.conv   # registers ConvBackbone/ConvDecoder with the factory

save(model, "conv_vae.model")
model = load("conv_vae.model")
```

`embkit.modules.conv` is not imported by `embkit.modules` itself (that would be a circular import, like `embkit.modules.mha`), so import it before `load` in a fresh session.

---

## Fusing several encoders: product of experts

With one encoder per data type, each produces its own Gaussian posterior for the same sample. `product_of_experts` combines them by precision: a confident view dominates, an uncertain one contributes little.

```python
from embkit.modules.fusion import product_of_experts

# mu_rna, logvar_rna, mu_adt, logvar_adt: (batch, latent_dim) from two encoders
mu_rna, logvar_rna = torch.randn(64, 16), torch.zeros(64, 16)
mu_adt, logvar_adt = torch.randn(64, 16), torch.zeros(64, 16)

mu, logvar = product_of_experts([mu_rna, mu_adt], [logvar_rna, logvar_adt])
```

- **Missing views:** leave them out of the lists for that batch. A single expert is returned unchanged.
- **Learned weighting:** pass `weights` of shape `(batch, n_experts)`, for example a softmax over the outputs of a small gating network. Each expert's precision is scaled by its weight (plus `weight_eps`, so no expert is ever dropped entirely).

```python
gate_logits = torch.randn(64, 2)                    # from your gating network
mu, logvar = product_of_experts([mu_rna, mu_adt], [logvar_rna, logvar_adt],
                                weights=torch.softmax(gate_logits, dim=1))
```

---

## Removing a nuisance signal: gradient reversal

To strip batch or site signal from an embedding, attach a classifier that predicts the batch from the latent and train the encoder to defeat it. `GradientReversal` is the identity going forward and flips the gradient going back, so one optimizer step does both:

```python
import torch.nn as nn
import torch.nn.functional as F
from embkit.models.vae import BaseVAE
from embkit.factory.layers import Layer
from embkit.losses import MSEVAELoss
from embkit.modules.grad_reverse import GradientReversal

n_features, n_batches = 200, 3
vae = BaseVAE(features=[f"g{i}" for i in range(n_features)], latent_dim=16,
              encoder_layers=[Layer(128, activation="relu")], decoder_layers=[Layer(128, activation="relu")])
reverse = GradientReversal(lambd=0.0)
batch_head = nn.Sequential(reverse, nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, n_batches))
recon_loss = MSEVAELoss()
optimizer = torch.optim.Adam(list(vae.parameters()) + list(batch_head.parameters()), lr=1e-3)

x = torch.randn(512, n_features)
batch_labels = torch.randint(0, n_batches, (512,))

n_epochs = 50
for epoch in range(n_epochs):
    reverse.lambd = min(1.0, epoch / 10)            # ramp up the adversary over the first 10 epochs
    out = vae(x)
    loss = (recon_loss(out.recon, x, out.mu, out.logvar).total
            + F.cross_entropy(batch_head(out.mu), batch_labels))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The batch head still learns to predict the batch; the encoder receives the reversed gradient and learns to hide it. Ramping `lambd` up from 0 lets the head become informative before the encoder starts fighting it. Check the result with a batch-mixing metric, and make sure cell-type structure survives: a strong adversary can erase biology that correlates with batch.

`grad_reverse(x, lambd)` is the functional form, for use outside `nn.Sequential`.

---

## Segmenting along an ordering: Gaussian HMM

When you expect runs of a few discrete states along an ordering — samples along pseudotime, bins along the genome, time points — an HMM turns a noisy continuous signal into a clean segmentation:

```python
from embkit.models.hmm import segment_by_order

latent = np.load("latent.npy")                      # (samples, dims), or a single score (samples,)
pseudotime = np.load("pseudotime.npy")              # one value per sample

states, probs, hmm = segment_by_order(latent, pseudotime, n_states=3, stickiness=0.95)
```

`states` and `probs` come back in the original sample order. State numbers are arbitrary labels; name them by inspecting `hmm.means`.

| Argument | Effect |
|----------|--------|
| `n_states` | number of segments to look for |
| `stickiness` | initial probability of staying in the same state; closer to 1 gives longer runs (all transition probabilities are then learned) |
| `n_iter` | optimisation steps (Adam, learning rate 0.05) |

For several sequences of equal length, fit `GaussianHMM` directly on a `(n_sequences, length, dims)` array:

```python
from embkit.models.hmm import GaussianHMM

tracks = np.random.default_rng(0).normal(size=(10, 200, 1))   # e.g. 10 samples x 200 genomic bins
model = GaussianHMM(n_states=3, dim=1).fit(tracks, n_iter=200)
paths = model.viterbi(tracks)        # (10, 200) most likely state per position
posterior = model.posterior(tracks)  # (10, 200, 3) state probabilities
```

Computation runs in float64, so the posterior probabilities come back as float64 tensors.

See the API pages for the [conv backbone and decoder](../api/layers/conv.md), [product of experts](../api/layers/fusion.md), [gradient reversal](../api/layers/grad_reverse.md) and the [Gaussian HMM](../api/models/hmm.md).
