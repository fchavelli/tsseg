# AMOC (At Most One Change)

Single change point detector that searches for the breakpoint minimising the
total sum of squared errors (SSE) on either side of the split. Foundational
building block for multi-change detectors such as Binary Segmentation and PELT.

## Key properties

- Type: change point detection
- Fully unsupervised (no hyperparameter for the number of segments)
- Detects at most one change point per call
- Univariate and multivariate
- O(n d) time, O(1) extra memory

## Implementation

Clean-room reimplementation of the classical SSE-based single change point
criterion. The API design and references are inspired by the R `changepoint`
package, but no R source code is reused.

- Origin: new code
- Reference package: https://github.com/rkillick/changepoint/ (GPL >= 2)

## Citation

```bibtex
@article{killick2012optimal,
  title   = {Optimal Detection of Changepoints with a Linear Computational Cost},
  author  = {Killick, Rebecca and Fearnhead, Paul and Eckley, Idris A.},
  journal = {Journal of the American Statistical Association},
  volume  = {107},
  number  = {500},
  pages   = {1590--1598},
  year    = {2012}
}
```
