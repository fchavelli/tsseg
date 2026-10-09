"""Download helper for reproduction tests.

Reproduction tests run a detector on a dataset used by the paper that
introduced it. The datasets are not shipped with tsseg: :func:`fetch`
downloads them on demand from their original source with pooch, checks their
SHA-256 digest and caches them, so that a test only runs on the exact bytes it
was written for.

When the source cannot be reached (no network), :func:`fetch` skips the
calling test instead of failing it. A wrong digest or an HTTP error (such as a
dead link) still fails.

This module belongs to the test suite, not to the public API of tsseg.

The cache lives in ``$TSSEG_DATA_DIR/reproduction`` when the environment
variable ``TSSEG_DATA_DIR`` is set, and in ``<user cache>/tsseg/reproduction``
otherwise (``pooch.os_cache``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

__all__ = ["REGISTRY", "DataFile", "cache_dir", "fetch"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ZIP_SUFFIXES = (".zip",)
_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


@dataclass(frozen=True)
class DataFile:
    """A remote file used by reproduction tests.

    Parameters
    ----------
    url : str
        Download URL at the original source of the dataset (``https`` is
        expected; ``http`` is accepted for local test servers).
    sha256 : str
        Lowercase hexadecimal SHA-256 digest of the downloaded file. The
        download is rejected when the digest differs.
    reference : str
        Where the dataset comes from: the paper that uses it and, if
        different, the paper or page that publishes it.
    license : str
        License or terms of use of the dataset, as stated by its source.
    fname : str or None, default=None
        File name in the cache. ``None`` lets pooch derive a unique name from
        the URL.
    unpack : bool, default=False
        Extract the archive (``.zip`` or ``.tar`` family, read from ``fname``
        or the URL) and return the paths of its members.

    Raises
    ------
    ValueError
        If the URL is not ``http(s)``, the digest is not a SHA-256 hex
        digest, ``reference`` or ``license`` is empty, or ``unpack`` is set
        for a file that is not a known archive.
    """

    url: str
    sha256: str
    reference: str
    license: str
    fname: str | None = None
    unpack: bool = False

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(f"URL must be http(s), got {self.url!r}.")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError(
                "sha256 must be the 64-character lowercase hex digest of the "
                f"file, got {self.sha256!r}."
            )
        if not self.reference.strip():
            raise ValueError("reference must name the source of the dataset.")
        if not self.license.strip():
            raise ValueError("license must state the terms of use of the dataset.")
        if self.unpack and self._archive_kind() is None:
            raise ValueError(
                f"Cannot unpack {self._name()!r}: expected one of "
                f"{_ZIP_SUFFIXES + _TAR_SUFFIXES}."
            )

    def _name(self) -> str:
        return self.fname or self.url.rsplit("/", 1)[-1].split("?", 1)[0]

    def _archive_kind(self) -> str | None:
        name = self._name().lower()
        if name.endswith(_ZIP_SUFFIXES):
            return "zip"
        if name.endswith(_TAR_SUFFIXES):
            return "tar"
        return None


#: Datasets available to reproduction tests, by name. Add an entry here (with
#: its source and license) rather than calling pooch directly from a test.
REGISTRY: dict[str, DataFile] = {
    "uci-glass": DataFile(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        sha256="dd67373f4baf2807345df02cbfef2093d342e61ad0d82a4fb79af43ef8ce449d",
        reference=(
            "Londschien, Bühlmann and Kovács, Random forests for change point "
            "detection, JMLR 2023 (Section 4.2); Evett and Spiehler, Glass "
            "Identification, UCI Machine Learning Repository, 1987"
        ),
        license="CC BY 4.0 (UCI Machine Learning Repository)",
        fname="glass.data",
    ),
}


def cache_dir() -> Path:
    """Return the directory where reproduction datasets are cached."""
    root = os.environ.get("TSSEG_DATA_DIR")
    if root:
        return Path(root) / "reproduction"
    import pooch

    return Path(pooch.os_cache("tsseg")) / "reproduction"


def fetch(data: str | DataFile) -> Path | list[Path]:
    """Download a dataset file if needed, check its digest, return its path.

    A cached file whose digest matches is returned without any network
    access; a cached file whose digest differs is downloaded again.

    Parameters
    ----------
    data : str or DataFile
        Name of an entry of :data:`REGISTRY`, or a :class:`DataFile`.

    Returns
    -------
    Path or list of Path
        Path of the downloaded file, or the sorted paths of the extracted
        members when ``unpack`` is set.

    Raises
    ------
    KeyError
        If ``data`` is a name that is not in :data:`REGISTRY`.
    ValueError
        If the downloaded file does not have the expected digest (raised by
        pooch, which then deletes the file).
    requests.HTTPError
        If the server answers with an HTTP error status.

    Notes
    -----
    A connection error or a timeout skips the calling test
    (``pytest.skip``).
    """
    import pooch
    import requests

    if isinstance(data, str):
        if data not in REGISTRY:
            raise KeyError(
                f"Unknown reproduction dataset {data!r}; "
                f"registered: {sorted(REGISTRY)}."
            )
        data = REGISTRY[data]

    processor = None
    if data.unpack:
        processor = pooch.Unzip() if data._archive_kind() == "zip" else pooch.Untar()

    try:
        result = pooch.retrieve(
            data.url,
            known_hash=f"sha256:{data.sha256}",
            fname=data.fname,
            path=cache_dir(),
            processor=processor,
            progressbar=False,
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        pytest.skip(f"Cannot download {data.url} (no network?): {exc}")
    if data.unpack:
        return sorted(Path(p) for p in result)
    return Path(result)
