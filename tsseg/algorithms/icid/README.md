# iCID: Isolation Distributional Kernel Change Interval Detection

This Python implementation adapts the MATLAB iCID algorithm maintained at [IsolationKernel/iCID](https://github.com/IsolationKernel/iCID). The original source code, authored by Yang Cao (Deakin University, Dec 2023 version 2.0), is licensed under the GNU General Public License version 3.0 (GPLv3) and serves as a demo for the method described in:

> Yang Cao, Ye Zhu, Kai Ming Ting, Flora Salim, Hong Xian Li, Luxing Yang, and Gang Li. Detecting Change Intervals with Isolation Distributional Kernel. *Journal of Artificial Intelligence Research*, 2024, 79:273–306. [[arXiv](https://arxiv.org/abs/2212.14630)]

## Algorithm overview

iCID detects change intervals in time series through four main steps:

1. **Distributional Kernel Transformation**: Each observation is mapped into a high-dimensional feature space by measuring proximity to randomly sampled subsets (`aNNEspace`).
2. **Dissimilarity Scoring**: The transformed series is divided into adjacent windows, and cosine similarity between their mean representations yields a dissimilarity score.
3. **Automatic `psi` Selection**: Multiple granularities (`psi`) are tested, selecting the one that minimizes approximate entropy over the score series.
4. **Adaptive Thresholding**: A statistical threshold is applied to highlight candidate change points.

## Usage

The `ICIDDetector` class follows the `tsseg.algorithms.base.BaseSegmenter` API.

```python
from tsseg.algorithms.icid.detector import ICIDDetector
import numpy as np

# Create a toy signal with a change point
signal = np.concatenate([np.random.rand(200, 2), np.random.rand(200, 2) + 0.5])

detector = ICIDDetector(window=50, alpha=0.5)
change_points = detector.fit_predict(signal)

print(f"Detected change points: {change_points}")
```

## Parameters

- `window` (int): Window length for dissimilarity computation.
- `alpha` (float): Sensitivity factor for thresholding; higher values reduce sensitivity.
- `t` (int): Number of iterations used to build the feature space.

## References

- Yang Cao et al., *Detecting Change Intervals with Isolation Distributional Kernel*, JAIR 2024. [[arXiv](https://arxiv.org/abs/2212.14630)]
- Original MATLAB release: [IsolationKernel/iCID](https://github.com/IsolationKernel/iCID) (GPLv3).
