"""Run every gallery example, offline, in the order of the documentation."""

import argparse
from importlib import import_module
from pathlib import Path

MODULES = [
    "jekyll",
    "storm_surge",
    "chloroplast",
    "climate_vocabulary",
    "mocap",
    "electricity",
    "speakers",
    "sleep",
]


def main():
    from examples.gallery._shared import run  # needs matplotlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for module in MODULES:
        study = import_module(f"examples.gallery.{module}").build()
        run(study, args.output)


if __name__ == "__main__":
    main()
