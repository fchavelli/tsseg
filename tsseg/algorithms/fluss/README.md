# FLUSS (Fast Low-cost Unipotent Semantic Segmentation)

Wrapper around the [stumpy](https://github.com/TDAmeritrade/stumpy) library's
FLUSS implementation. FLUSS uses the Matrix Profile to compute an Arc Curve
(corrected for the "regime change" bias), then detects semantic segmentation
boundaries at the valleys of this curve.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `n_cps`   | Number of change points to detect. |
| `m`       | Subsequence window length for the Matrix Profile. |
| `L`       | Subsequence window length for the Arc Curve. |

## Source

Uses [stumpy](https://github.com/TDAmeritrade/stumpy) (BSD 3-Clause license).

## References

```bibtex
@inproceedings{gharghabi2017matrix,
  title     = {Matrix Profile {VIII}: Domain Agnostic Online Semantic Segmentation
               at Superhuman Performance Levels},
  author    = {Gharghabi, Shaghayegh and Ding, Yifei and Yeh, Chin-Chia Michael
               and Kamgar, Kaveh and Ulanova, Liudmila and Keogh, Eamonn},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  pages     = {117--126},
  year      = {2017},
  doi       = {10.1109/ICDM.2017.21}
}
```
