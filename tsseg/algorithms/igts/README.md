# IGTS (Information Gain Temporal Segmentation)

Top-down segmentation algorithm that recursively splits the time series at the
point maximising information gain, treating each candidate segment as a
histogram of discretised values.

## Key properties

- Type: change point detection
- Semi-supervised (`k_max`) or unsupervised (step size controls granularity)
- Univariate and multivariate
- Uses cumulative-sum optimisation (Eq. 5-6) for O(m) segment entropy

## Changes from the aeon implementation

This implementation is adapted from the
[aeon toolkit](https://github.com/aeon-toolkit/aeon) (BSD 3-Clause) and
diverges from it in two ways, all justified by the original paper.

### 1. Normalise + complement augmentation applied to all data (Section 4.4)

The aeon code only augments **univariate** series (appending the complement
channel so that Shannon entropy varies across segments).  The paper's
Section 4.4 (Eq. 12-13) describes this transformation as a general
preprocessing step for **all** data — including multivariate — to handle
positively-correlated channels that would otherwise mask entropy differences.

We now apply `_augment_univariate` unconditionally: each channel is
normalised so values sum to 1 (Eq. 12), then its complement
(`max(c_i) − c_i`) is appended, doubling the channel count from *m* to
*2m* (Eq. 13).  On univariate data the behaviour is unchanged; on
multivariate data this fixes missed change points when channels are
positively correlated.

### 2. Cumulative-sum speed-up (Eq. 5-6)

The aeon code recomputes segment column sums from scratch for every
candidate at every iteration, making each IG evaluation O(m·n).
The paper (Eq. 5-6) proposes precomputing the cumulative sum
$F_i(t) = \sum_{j=1}^{t} c_{i_j}$ once, so that the sum over any segment
$[t_{j-1}, t_j)$ is obtained by a single subtraction:

$$p_{ji} = \frac{F_i(t_j) - F_i(t_{j-1})}{\sum_p F_p(t_j) - F_p(t_{j-1})}$$

This reduces each IG evaluation from O(m·n) to O(k·m) and yields a
measured ~5× speed-up on a 1 000-point, 4-channel series.

## Potential future implementation from paper

- Dynamic programming optimisation (Algorithm 2) — produces global optimum
  but O(kn²); not implemented yet.
- Automatic k estimation via knee-point detection (Section 4.5).

## Implementation

Adapted from the aeon toolkit.

- Origin: adapted from aeon
- Licence: BSD 3-Clause (aeon toolkit)

## Citation

```bibtex
@article{sadri2017information,
  title   = {Information Gain-based Metric for Recognizing Transitions in
             Human Activities},
  author  = {Sadri, Amin and Ren, Yongli and Salim, Flora D.},
  journal = {Pervasive and Mobile Computing},
  volume  = {38},
  pages   = {92--109},
  year    = {2017}
}
```
