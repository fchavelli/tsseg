# AutoPlait

**AutoPlait** automatically mines co-evolving time sequences by discovering
*regimes* (recurring dynamical patterns modelled as HMMs) and segmenting the
series accordingly.  It uses the **Minimum Description Length** (MDL) principle
to jointly determine the number of regimes and their boundaries — no prior
knowledge of the number of segments is required.

**Key properties:**

- Fully unsupervised (automatic model selection via MDL)
- Regime-based: each segment is assigned to a shared HMM regime
- Handles multivariate co-evolving sequences
- Requires a compiled C binary (see below)

---

## Implementation

The active implementation wraps the **original C code** by the paper's authors.
The C source lives under `c/autoplait/` at the repository root and must be
compiled before use:

```bash
cd c/autoplait && make clean autoplait
```

The Python wrapper (`autoplait_c.py`) writes data to a temp directory, invokes
the binary via `subprocess`, and parses the output files (`segment.*`,
`segment.labels`).

> Original C code: <https://sites.google.com/site/onlinesemanticsegmentation/>
>
> Author: Yasuko Matsubara (Kumamoto University)
>
> No explicit license file is provided with the original C code.

### Deprecated Python port

The `deprecated/` folder contains an incomplete pure-Python port based on
[hayato0311/autoplait-python](https://github.com/hayato0311/autoplait-python).
It depends on `hmmlearn` and was never validated — it is kept for reference only
and is **not** used by the detector.

---

## Citation

```bibtex
@inproceedings{matsubara2014autoplait,
    title     = {{AutoPlait}: Automatic Mining of Co-evolving Time Sequences},
    author    = {Matsubara, Yasuko and Sakurai, Yasushi and Faloutsos, Christos},
    booktitle = {Proceedings of the 2014 ACM SIGMOD International Conference
                 on Management of Data},
    pages     = {193--204},
    year      = {2014},
    doi       = {10.1145/2588555.2588556},
}
```
