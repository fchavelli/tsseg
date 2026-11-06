"""Recurring generation mixes in Great Britain, January 2024: CLaP."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import ClapDetector


def build():
    return Study(
        "electricity",
        "A windier generation mix returns after a gas-heavy interval",
        load("grid"),
        ["wind", "gas", "nuclear"],
        ["wind", "gas"],
        [
            (
                "CLaP",
                ClapDetector(
                    window_size=12,
                    distance="euclidean_distance",
                    n_jobs=1,
                    n_splits=3,
                    sample_size=256,
                    random_state=42,
                ),
            )
        ],
        ylabel="Generation share (%)",
        source="grid",
    )


if __name__ == "__main__":
    run(build())
