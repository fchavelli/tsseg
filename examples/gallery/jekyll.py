"""Narrative voice in Jekyll and Hyde: PELT on two pronoun rates."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import PeltDetector


def build():
    return Study(
        "jekyll",
        "Pronoun rates change where Lanyon’s narrative begins",
        load("novel"),
        ["first_person", "third_person"],
        ["first_person", "third_person"],
        [
            (
                "PELT",
                PeltDetector(
                    penalty=2, pen_scale="bic", min_size=5, jump=1, backend="python"
                ),
            )
        ],
        ylabel="Pronouns per 100 words",
        xlabel="Word position",
        truth="testimony",
        source="novel",
    )


if __name__ == "__main__":
    run(build())
