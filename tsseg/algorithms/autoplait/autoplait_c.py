import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


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
    """
    To run this method:
      1. download the adapted autoplait from https://sites.google.com/site/onlinesemanticsegmentation/
      2. place it into c/autoplait
      3. build it (make clean autoplait)
    """

    def __init__(self):
        """
        Initialize AutoPlait.
        """
        super().__init__()

    def _run_autoplait(self, ts, n_cps, name="autoplait_run"):
        """Run the AutoPlait binary and return one regime label per time point.

        The command line with a number of segments (``n_cps + 1``) is tried
        first; the bundled binary rejects it with a usage message, and the
        unguided command line is then run.

        Parameters
        ----------
        ts : np.ndarray
            Series of shape ``(n_samples,)`` or ``(n_samples, n_features)``.
        n_cps : int or None
            Number of change points to request, or ``None``.
        name : str, default="autoplait_run"
            Name of the temporary input file.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_samples,)``: the regime number of each
            time point (see :func:`_read_regime_labels`).
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

            np.savetxt(data_path, ts)
            np.savetxt(list_path, [data_path], fmt="%s")

            binary_path = autoplait_path / "autoplait"
            if not binary_path.exists():
                raise FileNotFoundError(
                    "AutoPlait executable not found. Expected at "
                    f"{binary_path}. Build it with `make autoplait` in {autoplait_path}."
                )

            candidate_commands: list[list[str]] = []
            if n_cps is not None:
                try:
                    candidate_commands.append(
                        [
                            str(binary_path),
                            str(n_features),
                            str(int(n_cps) + 1),
                            list_path,
                            tmp_path,
                        ]
                    )
                except (TypeError, ValueError):
                    pass

            candidate_commands.append(
                [
                    str(binary_path),
                    str(n_features),
                    list_path,
                    tmp_path,
                ]
            )

            candidate_commands.append(
                [
                    str(binary_path),
                    str(n_features),
                    list_path,
                    tmp_path,
                    str(n_samples),
                ]
            )

            last_error: RuntimeError | None = None
            for command in candidate_commands:
                result = subprocess.run(
                    command,
                    cwd=str(autoplait_path),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    break
                message = result.stderr or result.stdout
                if "usage" in message.lower() and len(command) > 4:
                    last_error = RuntimeError(message.strip())
                    continue
                last_error = RuntimeError(
                    "AutoPlait execution failed with return code "
                    f"{result.returncode}: {message}"
                )
            else:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("AutoPlait execution failed with an unknown error.")

            first_segment = next(Path(tmp_path).rglob("segment.0"), None)
            if first_segment is None:
                raise RuntimeError(
                    "AutoPlait ran but wrote no segment file (segment.0 missing)."
                )
            return _read_regime_labels(first_segment.parent, n_samples)
