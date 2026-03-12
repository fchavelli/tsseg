# VQ-TSS (Vector Quantized Time Series Segmentation)

State detection model that combines a VQ-VAE bottleneck with a contrastive
(InfoNCE) future-prediction objective to learn discrete state codes for
every timestep.  State boundaries emerge from transitions in the predicted
codes.

## Architecture

1. **Encoder** — stack of dilated residual 1-D convolutions (`ResBlock1D`)
   with exponentially growing dilation (receptive field ≈ 30 steps).
2. **Vector Quantiser** — EMA-updated codebook (`VectorQuantizerEMA`).
   Maps continuous latents to the nearest codebook entry.
3. **Predictor** — lightweight residual block that predicts the *next*
   continuous latent from the current quantised one.

## Training objective

$$\mathcal{L} = \mathcal{L}_\text{InfoNCE} + \mathcal{L}_\text{VQ} + \lambda\,\mathcal{L}_\text{smooth}$$

- **InfoNCE** — per-item contrastive loss between predicted and target
  latents, with temporal-neighbour masking to exclude trivially similar
  negatives.  Memory is $O(B \cdot T^2)$ instead of $O((BT)^2)$.
- **VQ** — commitment loss (EMA codebook, no codebook gradient).
- **Smooth** — MSE on consecutive quantised latents to discourage
  excessive code switching.

## Key properties

- Type: state detection
- Fully unsupervised
- Univariate and multivariate
- Requires PyTorch
- Fully convolutional → inference on the whole series in one pass

## Parameters of interest

| Parameter | Effect |
|-----------|--------|
| `num_embeddings` | Max number of discrete states (codebook size) |
| `smoothness_weight` | Higher → fewer code transitions |
| `contrastive_temperature` | Lower → sharper contrastive objective |
| `neg_temporal_margin` | Timesteps masked as easy negatives |
| `window_size` | Training context length |

## Implementation

- Origin: new code
- Encoder + VQ-EMA + predictor are in `network.py`
