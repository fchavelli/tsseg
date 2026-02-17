# AutoPlait

Automatic mining of co-evolving time sequences by discovering regimes
(recurring dynamical patterns modelled as HMMs) and segmenting the series
accordingly. Uses the Minimum Description Length (MDL) principle to jointly
determine the number of regimes and their boundaries.

## Key properties

- Type: state detection
- Semi-supervised (requires $n_cps$)
- Regime-based: each segment is assigned to a shared HMM regime
- Multivariate
- Requires a compiled C binary (`c/autoplait/` at the repository root)

## Implementation

Wraps the original C code by Yasuko Matsubara (Kumamoto University). The
`autoplait_c.py` module handles I/O with the binary. A `deprecated/` folder
contains an incomplete pure-Python port (based on hayato0311/autoplait-python,
depends on `hmmlearn`) kept for reference only.

- Origin: wrapper around original C code
- Source: https://sites.google.com/site/onlinesemanticsegmentation/
- Licence: no explicit licence provided with the original C code

## Citation

```bibtex
@inproceedings{matsubara2014autoplait,
  title     = {{AutoPlait}: Automatic Mining of Co-evolving Time Sequences},
  author    = {Matsubara, Yasuko and Sakurai, Yasushi and Faloutsos, Christos},
  booktitle = {Proceedings of the 2014 ACM SIGMOD International Conference
               on Management of Data},
  pages     = {193--204},
  year      = {2014}
}
```
