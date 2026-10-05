# Prophet Detector

Change points of the piecewise-linear trend of Facebook's Prophet forecaster.
A synthetic daily time index is generated to satisfy Prophet's datestamp
requirement.

## Key properties

- Type: change point detection
- Semi-supervised (`n_changepoints`) or unsupervised (`delta_threshold`)
- Univariate and multivariate (two strategies: per-channel ensembling or L2 norm)
- Requires `prophet` and `cmdstanpy`

## Implementation

Wrapper around the upstream `prophet` package. Prophet lets the rate of its
trend change at a set of potential change points, with a sparse (Laplace) prior
on the rate changes `delta`. The detector passes `n_candidates` evenly spaced
potential change points (without the first and last samples) and scores each
candidate by its fitted `|delta|` relative to the median `|delta|` over the
candidates. It returns:

- the `n_changepoints` strongest when this number is given;
- otherwise those whose score exceeds `delta_threshold`, a relative form of the
  extraction rule of TCPDBench (van den Burg & Williams, 2020).

The defaults (`changepoint_prior_scale=500`, `n_candidates=200`,
`tolerance=0.02`, `"l2"`) were calibrated on 24 series of TSB-SEG: Prophet's
forecasting default prior (0.05) spreads a level shift over a long smooth ramp,
whose kinks fall far from the change.

A level shift falling between two candidates shows as two opposite rate
changes on consecutive candidates, so two returned change points are at least
`tolerance` apart (greedy non-maximum suppression). Series longer than
`max_points` are reduced to block means before the fit, which bounds Prophet's
cost (it grows with the length times the number of candidates); the candidates
stay evenly spaced over the full series. Seasonalities are off by default (the
time index is synthetic). For multivariate input, `"ensembling"` fits one
model per channel and averages the scores of each candidate over the channels;
`"l2"` fits one model on the L2 norm of the channels.

- Origin: wrapper around facebook/prophet: https://github.com/facebook/prophet
- Licence: MIT (facebook/prophet)
