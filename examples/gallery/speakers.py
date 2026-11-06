"""Returning speakers in a constructed sequence of spoken digits: TICC."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import TiccDetector


def build():
    frame = load("speech")
    return Study(
        "speakers",
        "Three returning speakers in a constructed sequence",
        frame,
        [f"mfcc_{i:02}" for i in range(1, 13)],
        ["mfcc_01", "mfcc_02"],
        [
            (
                "TICC (3 states)",
                TiccDetector(
                    n_states=3, window_size=1, beta=20, maxIters=30, num_proc=1
                ),
            )
        ],
        ylabel="Mean MFCC",
        xlabel="Utterance index",
        truth="speaker",
        source="speech",
    )


if __name__ == "__main__":
    run(build())
