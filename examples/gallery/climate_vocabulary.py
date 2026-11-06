"""When four expressions entered climate research titles: one PELT fit each."""

from examples.gallery._data import load
from examples.gallery._shared import Study, run
from tsseg.algorithms import PeltDetector

# Context for reading the result, chosen after looking at the data; never fitted.
CONTEXT = [
    ("climate justice", 2009, "COP15, Copenhagen"),
    ("loss and damage", 2013, "Warsaw mechanism, COP19"),
    ("just transition", 2018, "Silesia declaration, COP24"),
    ("climate emergency", 2019, "UK Parliament declaration"),
]


def build():
    return Study(
        "climate_vocabulary",
        "Four expressions enter climate research at different times",
        load("climate_titles"),  # one group per expression, fitted separately
        ["log_rate"],
        ["rate"],
        [
            (
                "PELT",
                PeltDetector(
                    penalty=2, pen_scale="bic", min_size=3, jump=1, backend="python"
                ),
            )
        ],
        ylabel="Titles per 1,000 climate titles",
        xlabel="Publication year",
        events=CONTEXT,
        source="climate_titles",
    )


if __name__ == "__main__":
    run(build())
