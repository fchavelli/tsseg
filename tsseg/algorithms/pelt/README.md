# PELT (Pruned Exact Linear Time)

Wrapper around the vendored `ruptures.Pelt` solver. PELT applies dynamic
programming with an optimal pruning rule that reduces average-case complexity
to $O(n)$ while guaranteeing the same global optimum as the exact DP solver,
provided the cost function satisfies certain regularity conditions.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `penalty` | Penalty value controlling the trade-off between fit and number of segments (unsupervised). |
| `model`   | Cost function (`"l1"`, `"l2"`, `"rbf"`, etc.). |
| `min_size` | Minimum segment length. |

## Source

Vendored from the [ruptures](https://github.com/deepcharles/ruptures) library
(BSD 2-Clause license).

## References

```bibtex
@article{killick2012optimal,
  title   = {Optimal Detection of Changepoints with a Linear Computational Cost},
  author  = {Killick, Rebecca and Fearnhead, Paul and Eckley, Idris A.},
  journal = {Journal of the American Statistical Association},
  volume  = {107},
  number  = {500},
  pages   = {1590--1598},
  year    = {2012},
  doi     = {10.1080/01621459.2012.737745}
}
```
