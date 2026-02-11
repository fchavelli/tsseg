# VQ-TSS (Vector Quantized Time Series Segmentation)

PyTorch implementation of VQ-TSS. The model combines a VQ-VAE bottleneck
with an InfoNCE contrastive objective to learn discrete state codes for
each time step. State boundaries are derived from transitions in the
predicted codes.

Requires `torch`.