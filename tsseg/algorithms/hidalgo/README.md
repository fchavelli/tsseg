# Hidalgo Algorithm Modifications

This implementation of Hidalgo is based on the version from the [`aeon`](https://github.com/aeon-toolkit/aeon) library.

## Modifications

### Numerical Stability Fix in `sample_p`

**Issue:**
The original implementation in `_gibbs_sampling` -> `sample_p` performed probability calculations in the linear domain:
```python
frac = ((r1 / rmax) ** (c1[k] - 1)) * (
       ((1 - r1) / (1 - rmax)) ** (c1[K - 1] - 1)
)
```
When `c1[k]` (the count of points in cluster $k$ plus prior) becomes large, computing `x ** (c1[k] - 1)` often results in a floating-point overflow (`inf`) or underflow (`0.0`). This subsequently causes invalid values in scalar multiplication.

**Fix:**
The calculation has been moved to the log-domain to ensure numerical stability.
Instead of computing $A \times B$, we compute $\exp(\log(A) + \log(B))$.

The logic now handles:
1.  **Log-domain arithmetic**: 
    $$ \text{log\_frac} = (\text{exp}_1) \times (\ln(r_1) - \ln(r_{\text{max}})) + (\text{exp}_2) \times (\ln(1-r_1) - \ln(1-r_{\text{max}})) $$
2.  **Corner cases**: Checks for $r_1 \le 0$ or $r_1 \ge 1$ inside the log calculation to avoid math domain errors.
3.  **Comparisons**: The acceptance criterion `frac > r2` is replaced by `log_frac > np.log(r2)`.

This prevents `RuntimeWarning: overflow encountered in scalar power` and `FloatingPointError: invalid value encountered in scalar multiply` during Gibbs sampling on larger datasets.
