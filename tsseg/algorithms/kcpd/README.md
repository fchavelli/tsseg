# Kernel Change Point Detection (KCPD)

Wrapper around the vendored `ruptures.KernelCPD` solver. This method
formulates change point detection as a kernel-based optimisation problem,
using either dynamic programming (when `n_cps` is given) or a penalised
approach.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `n_cps`   | Fixed number of change points (semi-supervised). |
| `pen`     | Penalty value (unsupervised). |
| `kernel`  | Kernel type (`"linear"`, `"rbf"`, `"cosine"`). |

## Source

Vendored from the [ruptures](https://github.com/deepcharles/ruptures) library
(BSD 2-Clause license).

## References

```bibtex
@article{arlot2019kernel,
  title   = {A Kernel Multiple Change-point Algorithm via Model Selection},
  author  = {Arlot, Sylvain and Celisse, Alain and Harchaoui, Zaid},
  journal = {Journal of Machine Learning Research},
  volume  = {20},
  number  = {162},
  pages   = {1--56},
  year    = {2019}
}
```
