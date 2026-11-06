"""Storm surge at The Battery, New York: PELT on the non-tidal residual."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import PeltDetector


def build():
    return Study(
        "storm_surge",
        "Two storm episodes appear once the tide is removed",
        load("tides"),
        ["residual"],
        ["observed", "predicted", "residual"],
        [
            (
                "PELT",
                PeltDetector(
                    penalty=4, pen_scale="bic", min_size=12, jump=1, backend="python"
                ),
            )
        ],
        ylabel="Water level / residual (m)",
        source="tides",
    )


if __name__ == "__main__":
    run(build())
