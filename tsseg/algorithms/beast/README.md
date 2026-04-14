# BEAST (Bayesian Estimator of Abrupt change, Seasonality, and Trend)

Bayesian ensemble algorithm for change-point detection and time series
decomposition. BEAST decomposes a time series into trend, seasonality, and
abrupt changes using Bayesian model averaging (BMA) over a large space of
piecewise-linear trend models and piecewise-harmonic seasonal models. Instead
of selecting a single "best" model, BEAST combines all candidate models
weighted by their posterior probability, yielding robust change-point
detection with uncertainty quantification.

## Key properties

- Type: change point detection
- Unsupervised (probability-thresholded) or semi-supervised (`max_cps`)
- Univariate and multivariate (native `beast123`, L2 norm reduction, or per-channel ensembling)
- Detects both trend and seasonal change points
- Provides posterior probability of change-point occurrence at each time step
- Captures nonlinear trends via Bayesian model averaging
- Implemented in C with Python/R/Matlab bindings (fast MCMC sampling)

## Implementation

Wraps the Rbeast C library, vendorized in `c/Rbeast/`. Build with `make`
in that directory. A numpy 2.0 compatibility patch is included (upstream
requires `numpy<2`).

- Origin: Rbeast Python package
- Source: https://github.com/zhaokg/Rbeast
- Licence: GPL-2.0 (Python wrapper) — C core included
- Vendorized: `c/Rbeast/` (patched for numpy ≥ 2.0)

## Citation

```bibtex
@article{ZHAO2019111181,
    title = {Detecting change-point, trend, and seasonality in satellite time series data to track abrupt changes and nonlinear dynamics: A Bayesian ensemble algorithm},
    journal = {Remote Sensing of Environment},
    volume = {232},
    pages = {111181},
    year = {2019},
    issn = {0034-4257},
    doi = {https://doi.org/10.1016/j.rse.2019.04.034},
    url = {https://www.sciencedirect.com/science/article/pii/S0034425719301853},
    author = {Kaiguang Zhao and Michael A. Wulder and Tongxi Hu and Ryan Bright and Qiusheng Wu and Haiming Qin and Yang Li and Elizabeth Toman and Bani Mallick and Xuesong Zhang and Molly Brown},
    keywords = {Changepoint, Bayesian changepoint detection, Disturbance ecology, Breakpoint, Trend analysis, Time series decomposition, Bayesian model averaging, Disturbances, Nonlinear dynamics, Regime shift, Ensemble modeling, Time series segmentation, Phenology},
}
```
