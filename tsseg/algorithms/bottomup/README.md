# Bottom-Up Segmentation

Wrapper around the vendored `ruptures.BottomUp` solver. The bottom-up
algorithm starts from the finest possible segmentation and iteratively merges
adjacent segments that yield the smallest increase in approximation cost,
until a stopping criterion is met.
Complexity
is $O(nlog(n))$ where $n$ is the signal
length.

## Stopping criteria

Exactly one of the following must be specified:

| Parameter | Description |
|-----------|-------------|
| `n_cps`   | Fixed number of change points (supervised). |
| `penalty` | BIC/AIC-style penalty (unsupervised). |
| `epsilon`  | Maximum total approximation error. |

## Source

Vendored from the [ruptures](https://github.com/deepcharles/ruptures) library
(BSD 2-Clause license).

## References

```bibtex
@article{keogh2001online,
  title   = {An Online Algorithm for Segmenting Time Series},
  author  = {Keogh, Eamonn and Chu, Selina and Hart, David and Pazzani, Michael},
  journal = {Proceedings of the IEEE International Conference on Data Mining},
  pages   = {289--296},
  year    = {2001}
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
