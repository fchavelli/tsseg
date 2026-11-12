# SNLDS diagnostics

The scripts fit each series without labels; only the reference state count is
supplied. They are a small guided diagnostic, not a full benchmark or a
reproduction of a paper's training corpus.

```sh
python -m examples.diagnostics.snlds_audit --case mean-shifts --seed 0
python -m examples.diagnostics.snlds_audit --case bounce --data-seed 1 --seed 0
python -m examples.diagnostics.snlds_audit --case mocap --trial 1 --seed 0 \
  --params '{"feature_window": 128, "downsample": 6}'
```

Without `--steps`, each implementation uses its constructor's budget. Output
records actual parameters, input hashes, implementation fingerprints (excluding
docstrings/comments), scores and timings. Two predictions check repeatability.

Development used MoCap 0, TSSB 0 and the data-seed-0 synthetic tasks. Fixed
validation uses MoCap 1/2, TSSB 3/6, SKAB 3/6 and data-seed-1 synthetics. The two
candidate profiles are raw defaults and local moments with window 128 and
sampling interval 6. Both profiles are evaluated on all real cases, without
choosing a best profile per series after scoring.

Prepare `tssb3.npz`, `tssb6.npz`, `skab3.npz` and `skab6.npz`, containing `X`
(time by channels) and `y` (dense labels), using the stable benchmark order:

```python
from pathlib import Path
import numpy as np
from tsseg_exp.data_loading.loaders import load_dataset

destination = Path("/path/to/prepared-validation")
destination.mkdir(exist_ok=True)
for dataset in ("tssb", "skab"):
    X, y = load_dataset(dataset, data_root="/path/to/tsseg-exp/data")
    for i in (3, 6):
        x = np.asarray(X[i])
        if x.ndim == 1:
            x = x[:, None]
        np.savez(destination / f"{dataset}{i}.npz", X=x, y=y[i])
```

Compare an immutable pre-audit checkout at `e0b09b3` to the candidate:

```sh
python -m examples.diagnostics.snlds_validate \
  --baseline-root /path/to/pre-audit-tsseg \
  --candidate-root /path/to/candidate-tsseg \
  --data-dir /path/to/prepared-validation \
  --output /path/to/persistent-results
```

This runs 42 candidate fits and 12 baseline fits. Candidate seeds are 0/1/2;
baseline seed 0 runs on all cases, with 1/2 additionally on MoCap 1 and TSSB 3.
Single-run baseline cells cannot estimate seed variance. Four jobs run with
one thread each. Comparisons combine different budgets and implementations;
they are not isolated backend timings.

Results are written on every completion. `--resume` reuses completed logs only
when the source fingerprints, inputs and protocol match the saved manifest.
No successful or unsuccessful seed is dropped from the complete report.

```sh
python -m examples.diagnostics.snlds_report \
  --run-dir /path/to/persistent-results \
  --json-output examples/diagnostics/snlds_validation_results.json \
  --rst-output docs/_includes/snlds_validation.txt
```

The report rejects incomplete, duplicated or mixed-source runs and inconsistent
inputs. It aggregates seeds within each series before averaging series. The
earlier correction-only record, `snlds_audit_results.json`, remains separate
from the validation of the subsequent architecture/training adaptations.

## Inspecting failure mechanisms

The post-validation inspection deliberately selects failures. It diagnoses
optimization, representation and decoding; it is not a new benchmark or a
paper reproduction. `snlds_mechanism_results.json` preserves all ten selected
fits, including unsuccessful controls. Seed 0 is used except where stated.
Only the guided state count comes from labels during fitting. ARIs are scored
on the observation grid, including the sampling interval for moments.

```sh
python -m examples.diagnostics.snlds_mechanism \
  --case npz --input /path/to/prepared-validation/tssb3.npz \
  --output /path/to/mechanism/tssb3-raw-0
python -m examples.diagnostics.snlds_mechanism \
  --case mocap --trial 1 --params '{"feature_window":128,"downsample":6}' \
  --output /path/to/mechanism/mocap1-moments-0
python -m examples.diagnostics.snlds_mechanism \
  --case bounce --seed 0 --output /path/to/mechanism/bounce-raw-0
python -m examples.diagnostics.snlds_mechanism \
  --case bounce --seed 0 --skip-cluster-assignment \
  --output /path/to/mechanism/bounce-reconstruction-only-0
```

The other six fits are raw bouncing-ball seed 1, bouncing-ball seed 0 with
`{"init_steps":0}`, raw MoCap 1, TSSB 3 with moments, TSSB 3 with
`{"inference_type":"autoregressive"}` and TSSB 3 with
`{"n_train_steps":5000}`. All synthetic data use data seed 1, as in validation.

The script saves arrays and weights after reconstruction/cluster initialization
and after variational training. It reports reconstruction, regime occupancy,
latent/observed k-means, controlled removal of emission or transition factors,
and an ELBO estimate on 16 fixed training-context windows with four samples.
These ELBO estimates are comparable between stages of the same representation,
not across different representations or against held-out performance. Timings
include probe overhead and are not performance measurements.

`--skip-cluster-assignment` retains the same reconstruction training and its
random-window consumption, then restores the original regime dynamics, initial
means and transition network weights. Thus this control isolates the cluster
assignments. Merely setting `init_steps=0` also changes the training-window
sequence. The controlled pair in the saved report has exactly equal initial
encoder outputs and reconstructions.

```sh
python -m examples.diagnostics.snlds_mechanism_replay \
  --run-dir /path/to/mechanism/mocap1-moments-0 --samples 16
```

Replay verifies the implementation fingerprint and every saved weight path and
shape. It compares mean-trajectory Viterbi decoding with an average of smoothed
state probabilities under 16 continuous posterior draws and with encoding in
training-length windows. The latter concatenates the encoded windows before
global decoding; it is a context probe, not a proposed prediction algorithm.
Replay was run for raw TSSB 3 and raw/moments MoCap 1. None of these diagnostics
selects a production configuration from reference scores.
