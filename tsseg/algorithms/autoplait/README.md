# AutoPlait

Automatic mining of co-evolving time sequences by discovering regimes
(recurring dynamical patterns modelled as HMMs) and segmenting the series
accordingly. Uses the Minimum Description Length (MDL) principle to jointly
determine the number of regimes and their boundaries.

## Key properties

- Type: state detection
- Unsupervised: the number of segments and regimes is chosen by MDL
  (`n_cps` is deprecated and ignored, AutoPlait has no guided mode)
- Regime-based: each segment is assigned to a shared HMM regime
- Multivariate
- Needs numba (`tsseg[accelerators]`)

## Implementation

`_plait.py` translates to Python, compiled with numba, the original C code by
Yasuko Matsubara (version of June 18, 2014), file by file: `plait.c`, `cps.c`,
`segbox.c`, `baum.c`, `forbackward.c`, `kmeans.c`, `viterbi.c`, `hmmutils.c`
and `tool.c`. It reproduces the `rand()` generator of the GNU C library, so
that it draws the same random numbers as the C binary, and returns the same
segmentation as the binary built with GCC on Linux. The departures from the C
code, where it fails or reads uninitialised memory, are listed at the top of
`_plait.py`.

- Origin: translation of the original C code
- Source: https://www.dm.sanken.osaka-u.ac.jp/~yasuko/software.html
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
