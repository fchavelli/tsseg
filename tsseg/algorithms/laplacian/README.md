# Laplacian segmentation

State detection by spectral clustering of a recurrence graph of windows: the
Laplacian structural decomposition of McFee & Ellis (ISMIR 2014), designed for
the repeated sections of songs, applied to frames of a time series.

## Key properties

- Type: state detection
- Unsupervised (`n_states=None`) or guided (`n_states`)
- Univariate and multivariate (frame features of the channels concatenated)
- Cost: `O(N^2 d w + N^3)` time and `O(N^2)` memory for `N ≈ n / step`
  frames (dense graph and eigensolver); about 16 s and 0.9 GB for 56,000
  points with the defaults

## Implementation

- Origin: the recipe of the librosa gallery example "Laplacian segmentation"
  (ISC) and of msaf's `scluster` (MIT, defaults `rec_smooth = evec_smooth =
  9`), written for tsseg. The original repository of the paper's authors has
  no license and was not used.
- `_recurrence.py`: mutual k-NN recurrence affinity and diagonal median
  filter, ported from librosa 0.11.0 (`recurrence_matrix`, `timelag_filter`),
  dense, with identical outputs (tested against librosa when it is installed)
- `_louvain.py`: Louvain communities (Blondel et al., 2008), ported from
  NetworkX 3.5 (`louvain_communities`, undirected weighted case), with the
  same communities for a given seed (tested against NetworkX when it is
  installed)
- `laplacian.py`: frames, features, path graph, balanced graph, normalised
  Laplacian, eigenvectors, k-means, the two estimates of the number of states
- `detector.py`: `LaplacianDetector`
- Departures from the paper taken from the librosa example and msaf: cosine
  distance and `exp(-d / median)` instead of the Gaussian kernel of Sec. 2.3;
  `2 ceil(sqrt(N - 2e + 1))` neighbours instead of `1 + ceil(2 log2 N)` and an
  exclusion of `e` frames instead of 3 beats (Sec. 4.2); median filter of the
  weighted affinity instead of a majority vote over 17 frames on the binary
  graph (Eqs. 2, 8); path weights on the same features, median bandwidth
  (Eq. 9); median-filtered eigenvectors. The cosine ignores scale (states
  that differ only by amplitude with `raw`, levels of the same sign with
  `spec`, can look alike).
- tsseg choices (in neither): the filtered recurrence matrix is
  symmetrised, `(Rf + Rf.T) / 2`, because librosa's diagonal filter breaks its
  symmetry and `eigh` then decomposes a matrix that is not a Laplacian (the
  librosa example and msaf have this defect); the spectrum is checked, with a
  fallback to other LAPACK drivers. Frame features are log-magnitude spectra
  per channel (or raw windows) instead of beat-synchronous constant-Q and
  MFCC features; per-channel standardisation; fixed step in points, no cap on
  the number of frames. Unsupervised: `k = round(sqrt(k_L * k_S))`, Louvain
  communities of the recurrence graph (`2 ceil(sqrt(N))` neighbours, no
  diagonal filter, no path edges) and best cosine silhouette in feature
  space, selected on held-out TSB-SEG series (the paper's selector, Sec. 3.3,
  needs a minimal segment duration in seconds). Defaults `feature="spec"`,
  `window_size=128`, `step=16` declared before the TSB-SEG evaluation (592
  series): guided SMS 0.717 per series against 0.692 for CLaP.
- Reproduction: the paper evaluates songs (audio, constant-Q features), not
  reproducible here; `tests/reproduction/test_laplacian_mocap.py` recovers
  the TSB-SEG evaluation on the nine MoCap trials of `load_mocap` to the sixth
  decimal.
- Licence: AGPL-3.0 (tsseg); `_recurrence.py` ISC (librosa), `_louvain.py`
  BSD-3 (NetworkX); notices in `LICENSE`

## Citation

```bibtex
@inproceedings{mcfee2014analyzing,
  title     = {Analyzing Song Structure with Spectral Clustering},
  author    = {McFee, Brian and Ellis, Daniel P. W.},
  booktitle = {Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR)},
  pages     = {405--410},
  year      = {2014}
}

@article{blondel2008fast,
  title   = {Fast unfolding of communities in large networks},
  author  = {Blondel, Vincent D. and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
  journal = {Journal of Statistical Mechanics: Theory and Experiment},
  volume  = {2008},
  number  = {10},
  pages   = {P10008},
  year    = {2008},
  doi     = {10.1088/1742-5468/2008/10/P10008}
}
```
