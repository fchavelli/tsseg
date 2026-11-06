"""One night of polysomnography compared with expert sleep stages: HDP-HSMM."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import HdpHsmmDetector


def build():
    return Study(
        "sleep",
        "Unsupervised states compared with expert sleep stages",
        load("sleep"),
        [
            "ch0_delta",
            "ch0_theta",
            "ch0_alpha",
            "ch0_sigma",
            "ch0_beta",
            "ch2_delta",
            "emg",
        ],
        ["ch0_delta", "ch0_sigma"],
        [
            (
                "HDP-HSMM",
                HdpHsmmDetector(
                    n_max_states=6,
                    n_iter=30,
                    trunc=120,
                    dur_alpha=5,
                    dur_beta=5 / 19,
                    emission_strength=10,
                ),
            )
        ],
        ylabel="Log spectral energy",
        xlabel="Recording clock time",
        truth="stage",
        source="sleep",
    )


if __name__ == "__main__":
    run(build())
