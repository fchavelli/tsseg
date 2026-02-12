# Prophet Detector

Wraps Facebook's Prophet forecasting library to extract change points from its
internal piecewise-linear trend model. A synthetic daily time index is generated
to satisfy Prophet's datestamp requirement.

## Key properties

- Type: change point detection
- Semi-supervised (`n_changepoints`)
- Univariate and multivariate (two strategies: per-channel ensembling or L2 norm)
- Requires `prophet` and `cmdstanpy`

## Implementation

Wrapper around the upstream `prophet` package. The model is fitted with
`changepoint_range=1.0` so that change points can appear across the entire
signal. For multivariate input, the ensembling strategy fits one Prophet model
per channel and aggregates with a tolerance-based voting mechanism.

- Origin: wrapper around facebook/prophet: https://github.com/facebook/prophet
- Licence: MIT (facebook/prophet)
