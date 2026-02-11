# Dynamic Programming Segmentation (DynP)

Wrapper around the vendored `ruptures.Dynp` solver. This is the exact
optimal partitioning algorithm that finds the segmentation minimising the
total cost over all possible placements of `n_cps` change points. Complexity
is $O(CQn^2)$ where $C$ is the number of change points, $Q$, the complexity of calling the considered cost function on one sub-signal and $n$ the signal
length.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `n_cps`   | Number of change points (required — semi-supervised). |
| `model`   | Cost function (`"l1"`, `"l2"`, `"rbf"`, etc.). |
| `min_size` | Minimum segment length. |

## Source

Vendored from the [ruptures](https://github.com/deepcharles/ruptures) library
(BSD 2-Clause license).

## References

```bibtex
@article{auger1989algorithms,
  title   = {Algorithms for the Optimal Identification of Segment Neighborhoods},
  author  = {Auger, Ivan E. and Lawrence, Charles E.},
  journal = {Bulletin of Mathematical Biology},
  volume  = {51},
  number  = {1},
  pages   = {39--54},
  year    = {1989}
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
