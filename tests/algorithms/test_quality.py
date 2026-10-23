"""Detection quality on an easy series.

The contract tests of ``test_predict.py`` accept any well-formed output,
including no change point at all or a single state. The test of this module
asks for a minimal quality on the univariate series of the ``synthetic_data``
fixture (the 2-channel one for multivariate-only detectors): three Gaussian
segments, change points at 300 and 700 in 1000 points, means that differ by
2.4 to 6.5 standard deviations of the noise.

Each detector is built as in the generic suite (``ALGORITHM_OVERRIDES``),
with ``random_state=0`` when it takes a seed and defaults to ``None``. It
runs in guided mode when it has one (the true count is passed to the
constructor, so what is tested is where the change points or states fall),
unsupervised otherwise.

- Change point detectors: F1 score of at least 0.75 with a margin of 50
  points (5 % of the series), that is both change points found with at most
  one spurious change point.
- State detectors: adjusted Rand index of at least 0.8.

Detectors below this bar are listed in :data:`KNOWN_FAILURES`, with what they
returned when the entry was written. Their test is an ``xfail``, strict for
deterministic detectors: once one is fixed, its test fails until its entry
is removed.

The test is marked ``slow``; ``pytest -m "not slow"`` leaves it out.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.metrics import AdjustedRandIndex, F1Score

from .conftest import apply_supervision, make_instance

#: Minimal F1 score of a change point detector (margin of 50 points).
MIN_F1 = 0.75
#: Minimal adjusted Rand index of a state detector.
MIN_ARI = 0.8
#: Margin of the F1 score, as a fraction of the series length.
F1_MARGIN = 0.05

#: Detectors from which no quality is expected.
NO_QUALITY_EXPECTED: dict[str, str] = {
    "RandomDetector": "random baseline",
}

#: Detectors that return at most one change point: they are scored on the
#: first 700 points of the series, which hold the change point at 300 only.
AT_MOST_ONE_CHANGE_POINT = {"AmocDetector"}

#: Detectors below the bar, with the reason: what they returned on the
#: series, in the mode and with the constructor arguments of this test.
KNOWN_FAILURES: dict[str, str] = {
    # --- change point detectors ----------------------------------------
    "ChangeFinderDetector": (
        "[144, 306] guided; [306] unsupervised: the AR model predicts the "
        "change at 700, which leaves no peak near it"
    ),
    "ClaspDetector": (
        "[167, 305] guided with the suite's window_size=10, [206, 637] with "
        "the default: z-normalised windows miss mean shifts in noise "
        "([273, 679] with distance='euclidean_distance')"
    ),
    "FLUSSDetector": (
        "[485, 739] guided; none unsupervised (threshold 0.5): on z-normalized "
        "windows of white noise the mean shifts do not show, the corrected arc "
        "curve stays above 0.95 ([293, 692] guided with normalize=False)"
    ),
    "FLUSSDetector": "[485, 739] guided; [485] unsupervised",
    "ProphetDetector": (
        "[288, 308] guided; 14 change points unsupervised: the trend fits the "
        "shift at 300 as a ramp wider than tolerance (two change points), and "
        "the shift at 700 scores at the level of the noise"
    ),
    "ProphetDetector": "[288, 308] guided; 14 change points unsupervised",
    "TGLADDetector": (
        "unsupervised: [250] with the suite's window_size=100, stride=50, "
        "epochs=100; no change point with the defaults"
    ),
    "TSCP2Detector": (
        "[303, 308] after the suite's 2 epochs; [333, 348] after the default 100"
    ),
    # --- state detectors -----------------------------------------------
    "ClapDetector": (
        "ARI 0.26 (2 states) guided, ClaSP's change points [206, 637]: "
        "z-normalised windows miss mean shifts in noise (0.86, 3 states, with "
        "distance='euclidean_distance'); a single state unsupervised"
    ),
    "E2USDDetector": "ARI 0.35 (3 states) guided; 0.37 (6 states) unsupervised",
    "HidalgoDetector": (
        "multivariate series: ARI 0.01 (3 states) with the suite's n_iter=100; "
        "0.03 (4 states) with the default 1000"
    ),
    "HMMDetector": "a single state (deprecated detector)",
    "SNLDSDetector": (
        "ARI 0.19 (2 states) after the suite's 3 training steps; 0.97 after "
        "the default n_train_steps (110 s)"
    ),
    "Time2StateDetector": "ARI 0.22 (3 states) guided; 0.33 (4 states) unsupervised",
}

#: Detectors whose output varies from run to run without the
#: ``non_deterministic`` tag (TS-CP2: T29); their ``xfail`` is not strict.
_UNDECLARED_NON_DETERMINISTIC = {"TSCP2Detector"}


def _quality(instance, result: np.ndarray, data: dict) -> tuple[str, float, float]:
    """Return ``(metric name, score, threshold)`` for one prediction."""
    y, n = data["y"], data["y"].shape[0]
    if instance.get_tag("returns_dense"):
        score = F1Score(margin=F1_MARGIN).compute(
            data["change_points"].tolist(), result.tolist(), n_timepoints=n
        )["score"]
        return "F1", score, MIN_F1
    score = AdjustedRandIndex().compute(y, result)["score"]
    return "ARI", score, MIN_ARI


def _series(name: str, instance, synthetic_data) -> dict:
    """The series a detector is scored on."""
    key = "univariate" if instance.get_tag("capability:univariate") else "multivariate"
    data = synthetic_data[key]
    if name not in AT_MOST_ONE_CHANGE_POINT:
        return data
    end = int(data["change_points"][1])
    return {
        "X": data["X"][:end],
        "y": data["y"][:end],
        "change_points": data["change_points"][:1],
    }


@pytest.mark.slow
def test_detects_the_easy_segmentation(algorithm, synthetic_data, request):
    """Each detector finds the three segments of the easy series."""
    name, cls, ovr, instance = algorithm
    if name in NO_QUALITY_EXPECTED:
        pytest.skip(NO_QUALITY_EXPECTED[name])
    if name in KNOWN_FAILURES:
        strict = name not in _UNDECLARED_NON_DETERMINISTIC and not instance.get_tag(
            "non_deterministic", raise_error=False, tag_value_default=False
        )
        request.applymarker(
            pytest.mark.xfail(
                reason=KNOWN_FAILURES[name], strict=strict, raises=AssertionError
            )
        )

    data = _series(name, instance, synthetic_data)
    detector = make_instance(cls, ovr)
    params = detector.get_params(deep=False)
    if "random_state" in params and params["random_state"] is None:
        detector.set_params(random_state=0)

    X, y = data["X"], data["y"]
    guided = ovr.semi_supervised or detector.get_tag(
        "capability:semi_supervised", raise_error=False, tag_value_default=False
    )
    if guided:
        apply_supervision(detector, y)
        result = np.asarray(detector.fit_predict(X, y))
    else:
        result = np.asarray(detector.fit_predict(X))

    metric, score, threshold = _quality(detector, result, data)
    found = (
        f"change points {result.tolist()}"
        if detector.get_tag("returns_dense")
        else f"states found: {np.unique(result).size}"
    )
    mode = "guided" if guided else "unsupervised"
    assert score >= threshold, (
        f"{name} ({mode}): {metric} = {score:.2f} < {threshold} on the easy "
        f"series ({found}; true change points {data['change_points'].tolist()})"
    )
