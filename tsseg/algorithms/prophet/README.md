# Prophet Detector for tsseg

This directory wraps the upstream [`prophet`](https://github.com/facebook/prophet) package as an [`aeon`](https://www.aeon-toolkit.org/) compatible segmenter. It extracts changepoints from Prophet's internal trend model.

## Dependencies

Install Prophet from conda-forge (preferred) or pip:

```bash
conda install -c conda-forge prophet
# or
pip install prophet
```

Prophet depends on the Stan toolchain supplied by [`cmdstanpy`](https://cmdstanpy.readthedocs.io/). After installing the package, ensure the toolchain is available:

```bash
python -m cmdstanpy.install_cmdstan
```

## Usage

```python
import numpy as np
from tsseg.algorithms.prophet import ProphetDetector

signal = np.concatenate([
    np.random.normal(0.0, 0.1, size=120),
    np.random.normal(1.5, 0.1, size=100),
    np.random.normal(-0.5, 0.1, size=130),
])
series = signal[:, None]  # shape (n_timepoints, n_channels)

# Initialize with a fixed number of changepoints to detect
# Default strategy is "ensembling" for multivariate data
detector = ProphetDetector(n_changepoints=15, multivariate_strategy="ensembling", tolerance=2)
detector.fit(series, axis=0)
change_points = detector.predict(series, axis=0)
print(change_points)
```

## Implementation Details

The detector adapts the Prophet forecasting model for change point detection on generic time series data:

1.  **Multivariate Handling**: The detector supports two strategies for multivariate input:
    *   **Ensembling (Default)**: Fits a separate Prophet model on each dimension. Detected change points are aggregated using a voting mechanism. A `tolerance` parameter allows grouping change points that occur at slightly different times across dimensions (e.g., $t$ and $t \pm n$).
    *   **L2 Norm**: Computes the **L2 norm** (Euclidean norm) across dimensions at each time step to create a single representative signal ("energy"). This is faster but may lose directional information.
2.  **Temporal Indexing**: Since the input is a raw signal without timestamps, the detector generates a synthetic daily time index (`2000-01-01`, `2000-01-02`, ...) to satisfy Prophet's requirement for a `ds` (datestamp) column. This preserves the sequential order of observations.
3.  **Changepoint Extraction**: The model is fitted with `changepoint_range=1.0` to allow changes across the entire signal. Detected changepoints are retrieved from the model's internal state and mapped back to the integer indices of the original signal.

## Testing

A focused unit test (`tests/test_prophet_detector.py`) validates that the segmenter can fit and return changepoints when the optional dependencies are available.
