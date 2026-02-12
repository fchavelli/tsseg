# ESPRESSO (Entropy and ShaPe awaRe timE-Series SegmentatiOn)

Hybrid segmentation model for multi-dimensional time series that combines
entropy-based and temporal-shape-based features. Originally designed for
wearable sensor and IoT data as a preprocessing step for tasks such as human
activity recognition.

## Key properties

- Type: change point detection
- Semi-supervised (requires `n_segments`)
- Univariate and multivariate
- Combines matrix profile, information gain and semantic density

## Implementation

Pure-Python translation of the original MATLAB code. The MATLAB sources are
kept under `matlab/` for reference; the active implementation lives in
`python/`. A legacy MATLAB-engine wrapper (`detector_matlab.py`) is also
present but not used by default.

- Origin: translated from original MATLAB code
- Source: https://github.com/cruiseresearchgroup/ESPRESSO
- Licence: none provided in the original repository

### File layout

```
espresso/
  detector.py              EspressoDetector (BaseSegmenter wrapper)
  detector_matlab.py       Legacy MATLAB-engine wrapper (not used by default)
  matlab/                  Original MATLAB reference code (read-only)
    ESPRESSO_Script.m
    separateGreedyIG.m
    calculateSemanticDensityMatrix.m
    IGTS/
    MatrixProfile/
    utils/
  python/                  Pure-Python translation (active)
    ESPRESSO_Script.py     Entry point: espresso()
    separateGreedyIG.py    Greedy IG peak selection
    calculateSemanticDensityMatrix.py
    IGTS/
      IG_Cal.py            Information gain computation
      Sh_Entropy.py        Shannon entropy
    MatrixProfile/
      timeseriesSelfJoinFast.py   FFT-based matrix profile
    utils/
      Clean_TS.py          Normalise + mirror + cumsum
```

## Citation

```bibtex
@inproceedings{deldari2020espresso,
  title   = {{ESPRESSO}: Entropy and Shape Aware Time-Series Segmentation for
             Processing Heterogeneous Sensor Data},
  author  = {Deldari, Shohreh and Smith, Daniel V. and Sadri, Amin
             and Salim, Flora D.},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable
             and Ubiquitous Technologies (IMWUT)},
  volume  = {4},
  number  = {3},
  year    = {2020},
  doi     = {10.1145/3411832}
}
```
