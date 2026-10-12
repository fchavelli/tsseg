import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# The binary sets the length of the series to the number of lines of its input,
# counted by ``fgets`` into a buffer of ``BUFSIZE`` = 254 bytes
# (``c/autoplait/tool.c``, ``count_lines``; ``hmm.h``): a line of more than 252
# characters counts as two rows or more, and the binary then reads a longer
# series than the one it was given, padded with rows of zeros.
_MAX_LINE_LENGTH = 252


def _format_series(ts: np.ndarray) -> str:
    """Format ``ts`` as the binary's input: one row per line, as float32.

    The binary reads each value with ``fscanf("%f")`` into a C ``float``.
    ``%.9g`` writes the float32 value of each entry exactly, in at most 15
    characters, so that a row of up to 15 channels always fits in one line
    that the binary counts once.

    Parameters
    ----------
    ts : np.ndarray
        Series of shape ``(n_samples, n_features)``.

    Returns
    -------
    str
        The text of the input file.

    Raises
    ------
    ValueError
        If a row takes more than ``_MAX_LINE_LENGTH`` characters, which the
        binary would count as several rows.
    """
    buffer = io.StringIO()
    np.savetxt(buffer, np.asarray(ts, dtype=np.float32), fmt="%.9g")
    text = buffer.getvalue()
    longest = max((len(line) for line in text.splitlines()), default=0)
    if longest > _MAX_LINE_LENGTH:
        raise ValueError(
            f"AutoPlait: a row of this series takes {longest} characters, and the "
            f"binary counts a line of more than {_MAX_LINE_LENGTH} characters as "
            "several time points (c/autoplait/tool.c, count_lines): it would "
            "segment a longer series than this one. Series of up to 15 channels "
            f"always fit; this one has {ts.shape[1]}."
        )
    return text


def _check_rows_read(stdout: str, n_samples: int) -> None:
    """Raise if the binary reports a series length other than ``n_samples``.

    The binary prints ``duration: <number of rows read>`` on its standard
    output; nothing is checked if that line is missing.
    """
    match = re.search(r"^duration:\s*(\d+)", stdout, flags=re.MULTILINE)
    if match is not None and int(match.group(1)) != n_samples:
        raise RuntimeError(
            f"The AutoPlait binary read a series of {match.group(1)} time points "
            f"instead of {n_samples}."
        )


def _find_autoplait_dir() -> Path:
    """Locate the AutoPlait C directory, trying several strategies.

    Resolution order:
      1. ``AUTOPLAIT_DIR`` environment variable (explicit override).
      2. Relative to this source file: ``<package_root>/c/autoplait``
         (works when the repo is checked-out or editable-installed).
      3. ``autoplait`` on ``PATH`` — if found, return its parent dir.

    Returns the directory that should contain the ``autoplait`` binary.
    Raises ``FileNotFoundError`` if nothing is found.
    """
    # 1. Env var
    env = os.environ.get("AUTOPLAIT_DIR")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p

    # 2. Relative to source  (…/tsseg/algorithms/autoplait -> …/c/autoplait)
    package_root = Path(ABS_PATH).resolve().parents[2]
    relative = package_root / "c" / "autoplait"
    if relative.is_dir() and (relative / "autoplait").is_file():
        return relative

    # 2b. Also try the directory even if binary is missing (will raise later)
    if relative.is_dir():
        return relative

    # 3. Binary on PATH
    on_path = shutil.which("autoplait")
    if on_path:
        return Path(on_path).resolve().parent

    raise FileNotFoundError(
        "AutoPlait C directory not found. Tried:\n"
        f"  • $AUTOPLAIT_DIR (not set or invalid)\n"
        f"  • {relative} (not found)\n"
        f"  • 'autoplait' on PATH (not found)\n"
        "Set AUTOPLAIT_DIR=/path/to/c/autoplait or build the binary."
    )


def _read_regime_labels(out_dir: Path, n_samples: int) -> np.ndarray:
    """Read the binary's ``segment.<r>`` files into one label per time point.

    For each regime ``r``, the binary writes a file ``segment.<r>`` whose lines
    hold the first and the last index (inclusive) of one segment of that
    regime. The label of a time point is the number ``r`` of its regime, so
    that two segments of the same regime share a label. ``segment.labels`` is
    not read: it has one row per regime, and its second column is the path of
    the regime in the split tree (``"01"``, ``"00"``…), not a label.

    Parameters
    ----------
    out_dir : Path
        Output directory of the binary.
    n_samples : int
        Length of the series.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_samples,)``.

    Raises
    ------
    RuntimeError
        If a line is malformed, or if the segments do not cover each time point
        exactly once.
    """
    regime_files = sorted(
        (int(path.name.split(".")[-1]), path)
        for path in out_dir.glob("segment.*")
        if path.name.split(".")[-1].isdigit()
    )
    labels = np.full(n_samples, -1, dtype=int)
    for regime, path in regime_files:
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise RuntimeError(f"Unexpected line {line!r} in AutoPlait {path.name}")
            start, end = int(fields[0]), int(fields[1])
            if not 0 <= start <= end < n_samples:
                raise RuntimeError(
                    f"AutoPlait segment [{start}, {end}] of {path.name} lies outside "
                    f"a series of {n_samples} points."
                )
            if np.any(labels[start : end + 1] != -1):
                raise RuntimeError(
                    f"AutoPlait segment [{start}, {end}] of {path.name} overlaps "
                    "another segment."
                )
            labels[start : end + 1] = regime
    uncovered = np.flatnonzero(labels == -1)
    if uncovered.size:
        raise RuntimeError(
            f"AutoPlait segments leave {uncovered.size} time points without a "
            f"regime (first: {uncovered[0]})."
        )
    return labels


class AutoPlait:
    """Runner of the AutoPlait C binary of Matsubara et al.

    The binary is built from the original C code (version of June 18, 2014,
    https://www.dm.sanken.osaka-u.ac.jp/~yasuko/software.html), kept in
    ``c/autoplait/`` of the source repository: build it with ``make autoplait``
    there, or point ``AUTOPLAIT_DIR`` to the directory that holds it. It takes
    no number of segments or regimes; its command line is
    ``autoplait d list_file out_dir``.

    The modified binary of the FLOSS project page
    (https://sites.google.com/site/onlinesemanticsegmentation/) is not
    supported: its second argument is a minimum number of regimes, and it
    rejects the command line above.
    """

    def __init__(self):
        """
        Initialize AutoPlait.
        """
        super().__init__()

    def _run_autoplait(self, ts, name="autoplait_run"):
        """Run the AutoPlait binary and return one regime label per time point.

        The binary runs once, unguided: ``autoplait d list_file out_dir``.

        Parameters
        ----------
        ts : np.ndarray
            Series of shape ``(n_samples,)`` or ``(n_samples, n_features)``.
        name : str, default="autoplait_run"
            Name of the temporary input file.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_samples,)``: the regime number of each
            time point (see :func:`_read_regime_labels`).

        Raises
        ------
        ValueError
            If a row of the series does not fit on one line of the binary's
            input (see :func:`_format_series`).
        RuntimeError
            If the binary exits with a non-zero code, reads a series of another
            length, or writes no segment file.
        """
        # Raw name sometimes leads to errors with filesystem handling
        name = f"{hash(name)}"

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = os.path.join(temp_dir, f"AutoPlait-{name}/")
            autoplait_path = _find_autoplait_dir()

            os.makedirs(tmp_path, exist_ok=True)

            ts = np.asarray(ts, dtype=float)
            if ts.ndim == 1:
                ts = ts[:, np.newaxis]
            if ts.ndim != 2:
                raise ValueError(
                    "AutoPlait expects a 2D array with shape (n_samples, n_features)"
                )

            n_samples, n_features = ts.shape
            data_path = os.path.join(tmp_path, f"{name}.txt")
            list_path = os.path.join(tmp_path, "list")

            Path(data_path).write_text(_format_series(ts))
            np.savetxt(list_path, [data_path], fmt="%s")

            binary_path = autoplait_path / "autoplait"
            if not binary_path.exists():
                raise FileNotFoundError(
                    "AutoPlait executable not found. Expected at "
                    f"{binary_path}. Build it with `make autoplait` in {autoplait_path}."
                )

            result = subprocess.run(
                [str(binary_path), str(n_features), list_path, tmp_path],
                cwd=str(autoplait_path),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    "AutoPlait execution failed with return code "
                    f"{result.returncode}: {message}"
                )

            _check_rows_read(result.stdout, n_samples)
            first_segment = next(Path(tmp_path).rglob("segment.0"), None)
            if first_segment is None:
                raise RuntimeError(
                    "AutoPlait ran but wrote no segment file (segment.0 missing)."
                )
            return _read_regime_labels(first_segment.parent, n_samples)
