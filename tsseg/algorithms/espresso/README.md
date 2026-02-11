# ESPRESSO

**ESPRESSO** (Entropy and ShaPe awaRe timE-Series SegmentatiOn) is a hybrid
segmentation model for multi-dimensional time series that exploits both entropy
and temporal shape properties.  It has been used to extract meaningful temporal
segments from high-dimensional wearable sensor data, smart devices, and IoT data
as a preprocessing step for Human Activity Recognition (HAR), trajectory
prediction, gesture recognition, lifelogging, and smart cities.

> Shohreh Deldari, Daniel V. Smith, Amin Sadri, Flora D. Salim
>
> Paper: <https://arxiv.org/abs/2008.03230>
>
> Original repo: <https://github.com/cruiseresearchgroup/ESPRESSO/tree/master>
> 
> No license file is provided in the original repository.

---

## Python port

The original MATLAB implementation (from the repo above) is kept under `matlab/`
for reference.  A pure-Python translation lives in `python/` and is exposed via
`detector.py` through the `EspressoDetector` class.

### File layout

```
espresso/
├── detector.py              # EspressoDetector (BaseSegmenter wrapper)
├── detector_matlab.py       # Legacy MATLAB-engine wrapper
├── __init__.py
├── README.md
├── matlab/                  # Original MATLAB reference code (read-only)
│   ├── ESPRESSO_Script.m
│   ├── separateGreedyIG.m
│   ├── calculateSemanticDensityMatrix.m
│   ├── IGTS/
│   ├── MatrixProfile/
│   └── utils/
└── python/                  # Pure-Python translation (active)
    ├── ESPRESSO_Script.py   # Entry point: espresso()
    ├── separateGreedyIG.py  # Greedy IG peak selection
    ├── calculateSemanticDensityMatrix.py
    ├── IGTS/
    │   ├── IG_Cal.py        # Information gain computation
    │   └── Sh_Entropy.py    # Shannon entropy
    ├── MatrixProfile/
    │   └── timeseriesSelfJoinFast.py  # FFT-based matrix profile
    └── utils/
        └── Clean_TS.py      # Normalise + mirror + cumsum
```

---

## Citation

```bibtex
@inproceedings{deldari2020espresso,
    title     = {Entropy and ShaPe awaRe timE-Series SegmentatiOn for processing
                 heterogeneous sensor data},
    author    = {Deldari, Shohreh and Smith, Daniel V. and Sadri, Amin
                 and Salim, Flora D.},
    journal   = {Proceedings of the ACM on Interactive, Mobile, Wearable
                 and Ubiquitous Technologies (IMWUT)},
    volume    = {4},
    number    = {3},
    articleno = {77},
    year      = {2020},
    url       = {https://doi.org/10.1145/3411832},
    doi       = {10.1145/3411832},
}
```

    
