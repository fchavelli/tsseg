# Window-Based Segmentation

Wrapper around the vendored `ruptures.Window` solver. A sliding window of
fixed width scans the signal and computes a discrepancy (gain) score at each
position. Change points are selected at the peaks of this score profile.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `width`   | Width of the sliding window. |
| `n_cps` / `pen` / `epsilon` | Stopping criterion (at least one required). |
| `model`   | Cost function (`"l1"`, `"l2"`, `"rbf"`, etc.). |

## Source

Vendored from the [ruptures](https://github.com/deepcharles/ruptures) library
(BSD 2-Clause license).

## References

```bibtex
@incollection{basseville1993detection,
  title     = {Detection of Abrupt Changes: Theory and Application},
  author    = {Basseville, Mich{\`e}le and Nikiforov, Igor V.},
  year      = {1993},
  publisher = {Prentice Hall}
}

@article{truong2020selective,
  title   = {Selective review of offline change point detection methods},
  author  = {Truong, Charles and Oudre, Laurent and Vayer, Nicolas},
  journal = {Signal Processing},
  volume  = {167},
  pages   = {107299},
  year    = {2020},
  doi     = {10.1016/j.sigpro.2019.107299}
}
```
