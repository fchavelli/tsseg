# Cachalot

`CachalotDetector` implements the Lasso-candidate and reduced
dynamic-programming method of Harchaoui and Lévy-Leduc. The implementation is
an independent reimplementation from the paper; it does not translate or
include original code. No original source repository is identified in the
paper, and the NeurIPS version links no code.

The paper supplied for this implementation is [*Catching Change-points with
Lasso*](https://papers.neurips.cc/paper_files/paper/2007/file/e5841df2166dd424a57127423d276bbe-Paper.pdf).
The journal version is [*Multiple Change-Point Estimation With a Total
Variation Penalty*](https://doi.org/10.1198/jasa.2010.tm09181).

The paper reports a toy example's Lasso candidates and objective values, but
does not provide its noise realization or random seed. The implementation's
model-selection rule is tested against those published objective values; the
full toy realization and the Monte Carlo tables cannot be reproduced exactly
from the published inputs.

The implementation is original tsseg code under the repository's
AGPL-3.0-only license. It contains no copied third-party code.

## References

```bibtex
@inproceedings{harchaoui2007catching,
  title     = {Catching Change-points with Lasso},
  author    = {Harchaoui, Zaid and L{\'e}vy-Leduc, C{\'e}line},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {20},
  year      = {2007}
}

@article{harchaoui2010multiple,
  title   = {Multiple Change-Point Estimation With a Total Variation Penalty},
  author  = {Harchaoui, Zaid and L{\'e}vy-Leduc, C{\'e}line},
  journal = {Journal of the American Statistical Association},
  volume  = {105},
  number  = {492},
  pages   = {1480--1493},
  year    = {2010},
  doi     = {10.1198/jasa.2010.tm09181}
}
```
