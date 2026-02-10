import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

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
        """
        Executes the external AutoPlait C program.
        """
        # Raw name sometimes leads to errors with filesystem handling
        name = f"{hash(name)}"

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = os.path.join(temp_dir, f"AutoPlait-{name}/")
            package_root = Path(ABS_PATH).resolve().parents[2]
            autoplait_path = package_root / "c" / "autoplait"

            os.makedirs(tmp_path, exist_ok=True)
            if not autoplait_path.exists():
                raise FileNotFoundError(
                    "AutoPlait binary directory not found. Expected at "
                    f"{autoplait_path}. Ensure the C implementation is built via"
                    " `make clean autoplait`."
                )

            ts = np.asarray(ts, dtype=float)
            if ts.ndim == 1:
                ts = ts[:, np.newaxis]
            if ts.ndim != 2:
                raise ValueError("AutoPlait expects a 2D array with shape (n_samples, n_features)")

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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
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

            tmp_path_obj = Path(tmp_path)

            segment_dir: Path | None = None
            for candidate in tmp_path_obj.rglob("segment.labels"):
                segment_dir = candidate.parent
                break
            if segment_dir is None:
                for candidate in tmp_path_obj.rglob("segment.0"):
                    segment_dir = candidate.parent
                    break

            if segment_dir is None:
                return np.array([], dtype=int), np.array([], dtype=int)

            segment_files = sorted(
                (
                    path
                    for path in segment_dir.glob("segment.*")
                    if path.name.split(".")[-1].isdigit()
                ),
                key=lambda path: int(path.name.split(".")[-1]),
            )

            found_cps: list[float] = []
            for segment_file in segment_files:
                pred = np.loadtxt(segment_file, dtype=np.float64)
                if pred.size == 0:
                    continue
                if pred.ndim == 1:
                    pred = np.atleast_1d(pred)
                    if pred.size == 1:
                        continue
                    pred = [pred[1]]
                elif pred.ndim == 2:
                    pred = pred[:, 1]
                else:
                    raise RuntimeError(f"Unexpected AutoPlait segment format in {segment_file}")
                found_cps.extend(pred)

            labels_path = segment_dir / "segment.labels"
            found_labels: list[int] = []
            if labels_path.exists():
                with labels_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip().split("\t\t")
                        if len(line) > 1:
                            try:
                                found_labels.append(int(line[1]))
                            except (TypeError, ValueError):
                                found_labels.append(0)
                        else:
                            found_labels.append(0)

        if found_cps:
            found_cps_array = np.array(found_cps, dtype=int)
            found_cps_array = np.sort(found_cps_array)
            if found_cps_array.size > 0:
                found_cps_array = found_cps_array[:-1]
        else:
            found_cps_array = np.array([], dtype=int)

        if found_labels:
            found_labels_array = np.array(found_labels, dtype=int)
        else:
            found_labels_array = np.array([], dtype=int)

        return found_cps_array, found_labels_array
