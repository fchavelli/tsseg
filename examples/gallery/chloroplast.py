"""GC content along a chloroplast genome: PELT, checked against rRNA genes."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import PeltDetector


def build():
    return Study(
        "chloroplast",
        "Two GC-rich regions contain the ribosomal RNA genes",
        load("dna"),
        ["gc"],
        ["gc"],
        [
            (
                "PELT",
                PeltDetector(
                    penalty=2, pen_scale="bic", min_size=5, jump=1, backend="python"
                ),
            )
        ],
        ylabel="GC (%)",
        xlabel="Reference position (base pairs)",
        source="dna",
    )


if __name__ == "__main__":
    run(build())
