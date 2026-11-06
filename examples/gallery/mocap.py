"""Human activities in motion capture (CMU subject 86, trial 01): TICC."""

import numpy as np
import pandas as pd

from examples.gallery._shared import Study, run
from tsseg.algorithms import TiccDetector
from tsseg.data.datasets import load_mocap

FRAME_RATE = 120  # frames per second
CHANNELS = ["right_upper_arm", "left_upper_arm", "right_thigh", "left_thigh"]
# Action names from the annotation of Barbič et al. (2004); never fitted.
ACTIVITIES = ["walking", "jumping forward", "punching", "leg kicking"]


def frame():
    X, y = load_mocap(trial=0)  # recording 86_01, bundled with tsseg
    data = pd.DataFrame(X, columns=CHANNELS)
    data.insert(0, "x", np.arange(len(X)) / FRAME_RATE)
    data.insert(1, "label", np.arange(len(X)).astype(str))
    data["activity"] = y
    data["activity_name"] = [ACTIVITIES[k] for k in y]
    return data


def build():
    return Study(
        "mocap",
        "Four activities recognised in a motion-capture recording",
        frame(),
        CHANNELS,
        ["right_upper_arm", "right_thigh"],
        [
            (
                "TICC (4 states)",
                # Default settings of the TSB-SEG benchmark, not tuned here.
                TiccDetector(
                    n_states=4,
                    window_size=3,
                    lambda_parameter=1e-3,
                    beta=3000,
                    maxIters=100,
                    threshold=1e-3,
                    num_proc=1,
                ),
            )
        ],
        ylabel="Rotation angle (degrees)",
        xlabel="Time (s)",
        truth="activity",
    )


if __name__ == "__main__":
    run(build())
