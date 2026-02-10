# TIRE

**TIRE** is an autoencoder-based change point detection algorithm for time series data that uses a TIme-Invariant Representation (TIRE). More information can be found in the paper *Change Point Detection in Time Series Data using Autoencoders with a Time-Invariant Representation*, published in *IEEE Transactions on Signal Processing* in 2021. 

The authors of this paper are:

- [Tim De Ryck](https://math.ethz.ch/sam/the-institute/people.html?u=deryckt) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven; now [SAM](https://math.ethz.ch/sam), Dept. Mathematics, ETH Zürich)
- [Maarten De Vos](https://www.esat.kuleuven.be/stadius/person.php?id=203) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven and Dept. Development and Regeneration, KU Leuven)
- [Alexander Bertrand](https://www.esat.kuleuven.be/stadius/person.php?id=331) ([STADIUS](https://www.esat.kuleuven.be/stadius/), Dept. Electrical Engineering, KU Leuven)

All authors are affiliated to [LEUVEN.AI - KU Leuven institute for AI](https://ai.kuleuven.be). Note that work based on TIRE should cite our paper: 

    @article{deryck2021change,
    title={Change Point Detection in Time Series Data using Autoencoders with a Time-Invariant Representation},
    author={De Ryck, Tim and De Vos, Maarten and Bertrand, Alexander},
    journal={IEEE Transactions on Signal Processing},
    year={2021},
    publisher={IEEE}
    }

## Abstract

*Change point detection (CPD) aims to locate abrupt property changes in time series data. Recent CPD methods demonstrated the potential of using deep learning techniques, but often lack the ability to identify more subtle changes in the autocorrelation statistics of the signal and suffer from a high false alarm rate. To address these issues, we employ an autoencoder-based methodology with a novel loss function, through which the used autoencoders learn a partially time-invariant representation that is tailored for CPD. The result is a flexible method that allows the user to indicate whether change points should be sought in the time domain, frequency domain or both. Detectable change points include abrupt changes in the slope, mean, variance, autocorrelation function and frequency spectrum. We demonstrate that our proposed method is consistently highly competitive or superior to baseline methods on diverse simulated and real-life benchmark data sets. Finally, we mitigate the issue of false detection alarms through the use of a postprocessing procedure that combines a matched filter and a newly proposed change point score. We show that this combination drastically improves the performance of our method as well as all baseline methods.*

## Goal

More concretely, the goal of TIRE is the following. Given raw time series data, TIRE returns for each time stamp of the time series a change point score. This score reflects the probability that there is a change point at (or near) the corresponding time stamp. Note that the absolute value of this change point score has no meaning. It is then common practice to discard the change point for which the change point score is below some user-defined treshold. For more information on how the change point scores are obtained we refer to our paper. 

Detectable change points include abrupt changes in: 
- Mean
- Slope
- Variance
- Autocorrelation
- Frequency spectrum

## Guidelines

First install all required packages. The aeon integration of TIRE included in this repository now relies on [PyTorch](https://pytorch.org/) instead of TensorFlow/Keras. Ensure that `torch` is available in your Python environment. We provided a Jupyter notebook `TIRE_example_notebook.ipynb` that demonstrates how the TIRE change point scores can be obtained from the raw time series data. In addition, the change points obtained by TIRE are compared in the notebook to the ground truth both visually and through the calculation of the AUC score. Alternatively, you can run `main.py` to obtain a txt-file containing the change point scores. 

### Using the aeon-compatible detector

The file `detector.py` wraps the core TIRE pipeline in the `TireDetector` class, which inherits from `tsseg.algorithms.base.BaseSegmenter`. This allows you to use the algorithm in the same way as the other detectors in the `tsseg.algorithms` namespace:

```python
import numpy as np
from tsseg.algorithms import TireDetector

# toy signal with two changes
rng = np.random.default_rng(0)
segments = [rng.normal(-0.5, 0.2, size=120), rng.normal(0.8, 0.2, size=100), rng.normal(-0.1, 0.2, size=120)]
series = np.concatenate(segments)[:, None]

detector = TireDetector(window_size=32, n_segments=3, max_epochs=10, patience=3, domain="TD")
detector.fit(series)
change_points = detector.predict(series)
print(change_points)
```

The detector returns a numpy array with the detected change point indices. Setting `n_segments` chooses how many change points to retain (the algorithm selects the most prominent `n_segments - 1` peaks). When `n_segments` is left unspecified, all peaks above a default prominence threshold are returned.

## Contact

In case of comments or questions, please contact me at <tim.deryck@math.ethz.ch>. 

## Remark

The original repository do not mention any License, authors granted use of this code in our work.