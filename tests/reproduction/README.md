# Reproduction tests

A reproduction test checks that a detector of tsseg reproduces a result
published for its method, on a dataset that the publication uses. These tests
need data that tsseg does not ship and can take minutes, so they never run in
CI: they are run on demand, before a release or after a change to the
detector.

## Running

```bash
python -m pytest -m reproduction                       # all reproduction tests
python -m pytest -m reproduction -k ClaspDetector      # one detector
python -m pytest -m "reproduction or not reproduction" # the whole suite
```

The `reproduction` marker is declared in `pyproject.toml`, and the default
options (`addopts`) contain `-m "not reproduction"`: a plain `pytest`, as in
CI, deselects these tests. A `-m` given on the command line replaces the
default one. Overriding the options (`-o addopts=""`, as for a coverage run)
also drops the deselection: add `-m "not reproduction"` back.

## Data

Datasets are downloaded on demand from their original source, never
committed to the repository nor included in the package. `_data.py` provides
the helper:

```python
from ._data import fetch

path = fetch("dataset-name")  # entry of REGISTRY in _data.py
```

- Each file is pinned by its SHA-256 digest. A download with another digest
  is rejected; a cached file whose digest differs is downloaded again.
- Files are cached in `$TSSEG_DATA_DIR/reproduction` if `TSSEG_DATA_DIR` is
  set, else in the user cache (`~/.cache/tsseg/reproduction` on Linux). A
  cached file is used without network access.
- Without network, `fetch` skips the test. An HTTP error (dead link) or a
  wrong digest fails it.
- Archives (`.zip`, `.tar.*`) are extracted with `unpack=True`; `fetch` then
  returns the sorted paths of their members.

To add a dataset, add a `DataFile` to `REGISTRY`:

```python
REGISTRY["tssb"] = DataFile(
    url="https://…",  # original source, https, a fixed version (tag or commit)
    sha256="…",  # python -c "import pooch; print(pooch.file_hash('file'))"
    reference="Paper that uses it; paper or page that publishes it",
    license="Terms of use stated by the source",
    unpack=True,
)
```

Point the URL at an immutable version of the file (a release asset, a tagged
or commit URL): a file that changes upstream breaks the digest. Check that
the terms of use of the dataset allow downloading it for testing; the test
suite downloads from the source and redistributes nothing.

## Writing a reproduction test

1. **Dataset**: one used by the publication of the method (not a convenient
   substitute), fetched with `fetch`.
2. **Marker**: `@pytest.mark.reproduction` on the test or the module
   (`pytestmark = pytest.mark.reproduction`).
3. **Reference value**: the number reported in the publication (cite the
   table or figure), or the output of the authors' code if the publication
   reports none and their license allows running it.
4. **Protocol**: follow the publication (preprocessing, hyperparameters,
   supervision, metric and its margin) and state every deviation.
5. **Tolerance**: justified in the docstring (randomness, rounding of the
   published value, differences of protocol).
6. **Docstring**: publication value, value obtained by tsseg, protocol,
   probable causes of the gap.
7. **Synthetic twin**: a fast test that runs in CI on synthetic data, on the
   same detector and code path, with a known answer. It catches regressions
   between two runs of the reproduction test.

Name the files `test_<detector>.py` in this directory, for example:

```python
"""ClaSP on TSSB, as in the ClaSP publication."""

import pytest

from tsseg.algorithms import ClaspDetector

from ._data import fetch

pytestmark = pytest.mark.reproduction


def test_clasp_tssb_covering():
    """Publication (Table k): mean Covering a; tsseg: b; tolerance t, because ..."""
    files = fetch("tssb")
    ...
```

`test_data_helper.py` tests the helper itself (offline, with a local HTTP
server); it is not marked and runs in CI.
