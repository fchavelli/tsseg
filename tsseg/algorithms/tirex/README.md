# TiRex

** /!\ Under development /!\ **

Bridge module that re-exports TiRex-based segmenters from the sibling
`tsseg-tirex` project. At import time the module dynamically adds the required
paths and exposes the following detector classes:

- `TirexHiddenCPD`, `TirexCosineCPD`, `TirexL2CPD`
- `TirexHiddenState`, `TirexCosineState`, `TirexL2State`
- `TirexHiddenCluster`, `TirexCosineCluster`, `TirexL2Cluster`
- `TirexEmbedding`
- `TirexSegmenter`, `TirexEmbeddingSegmenter`, `TirexClusterSegmenter`

If `tsseg-tirex` is not installed or not found, the import is silently skipped
and `__all__` is empty.

## Key properties

- Type: change point detection, state detection and clustering (depending on variant)
- Univariate and multivariate
- Requires the `tsseg-tirex` package and its dependencies

## Implementation

- Origin: re-export from `tsseg-tirex` (separate project)
