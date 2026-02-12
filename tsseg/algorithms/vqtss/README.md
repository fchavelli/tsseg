# VQ-TSS (Vector Quantized Time Series Segmentation)

State detection model that combines a VQ-VAE bottleneck with an InfoNCE
contrastive objective to learn discrete state codes for each time step. State
boundaries are derived from transitions in the predicted codes.

## Key properties

- Type: state detection
- Fully unsupervised
- Univariate and multivariate
- Requires PyTorch

## Implementation

New implementation. The VQ-VAE and predictor networks are in `network.py`.

- Origin: new code
