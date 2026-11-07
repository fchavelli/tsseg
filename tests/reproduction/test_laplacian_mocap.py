"""Reproduction of the TSB-SEG evaluation of Laplacian segmentation on MoCap.

Publication: B. McFee and D. P. W. Ellis, *Analyzing song structure with
spectral clustering*, ISMIR 2014. The paper evaluates boundary and label
agreement on songs (Beatles-TUT, SALAMI) from beat-synchronous constant-Q and
MFCC features of the audio. Neither the audio (not redistributable) nor an
audio front end is part of tsseg, so the paper's tables are not reproduced.

Reference instead: the evaluation that led to this detector, on the 592
series of TSB-SEG, with the librosa (``recurrence_matrix``, ``timelag_filter``)
and NetworkX (``louvain_communities``) functions that tsseg re-implements.
Its MoCap dataset is the nine trials embedded in tsseg (same lengths, same
estimated numbers of states). Protocol: per-channel standardisation, default
detector (``feature="spec"``, ``window_size=128``, ``step=16``), seed 0;
guided = true number of states; unsupervised = the default rule. Score: SMS
with the TSB-SEG weights (delay 0.1, transition 0.3, isolation 0.8, missing
0.5).

Result (2026-11-07): identical labels on the nine trials, so identical SMS:
mean 0.8386 guided and 0.8220 unsupervised, as in the reference. Tolerance
1e-4 per series (scores agree to the sixth decimal on the reference
machine); a different BLAS can move a k-means tie, which this test would
report. No data are downloaded: about 15 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg import load_mocap
from tsseg.algorithms import LaplacianDetector
from tsseg.metrics.state_detection import StateMatchingScore

pytestmark = pytest.mark.reproduction

SMS_WEIGHTS = {"delay": 0.1, "transition": 0.3, "isolation": 0.8, "missing": 0.5}

#: Reference SMS per MoCap trial (0 to 8): guided, unsupervised, and the
#: number of states chosen by the unsupervised rule.
REFERENCE = {
    0: (0.863670, 0.863670, 4),
    1: (0.812081, 0.819358, 7),
    2: (0.749021, 0.805257, 5),
    3: (0.859044, 0.768921, 4),
    4: (0.770721, 0.758735, 10),
    5: (0.797977, 0.797977, 5),
    6: (0.900472, 0.900472, 4),
    7: (0.954441, 0.843045, 5),
    8: (0.840417, 0.840417, 3),
}


@pytest.mark.parametrize("trial", sorted(REFERENCE))
def test_mocap_trial_matches_the_reference(trial):
    X, y = load_mocap(trial=trial)
    sms_guided, sms_unsupervised, k_unsupervised = REFERENCE[trial]
    metric = StateMatchingScore(weights=SMS_WEIGHTS)

    guided = LaplacianDetector(n_states=len(np.unique(y)), random_state=0)
    score = metric.compute(y, guided.fit_predict(X))["score"]
    assert score == pytest.approx(sms_guided, abs=1e-4)

    unsupervised = LaplacianDetector(random_state=0)
    score = metric.compute(y, unsupervised.fit_predict(X))["score"]
    assert unsupervised.n_states_ == k_unsupervised
    assert score == pytest.approx(sms_unsupervised, abs=1e-4)
