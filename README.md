<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/fchavelli/tsseg/main/assets/logo_dark.png">
    <img src="https://raw.githubusercontent.com/fchavelli/tsseg/main/assets/logo_light.png" alt="tsseg logo" width="160">
  </picture>
</p>

<p align="center">
  <b>Time series segmentation in Python</b><br>
  Change point detection and state detection: 30+ algorithms, one scikit-learn-style API
</p>

<p align="center">
  <a href="https://doi.org/10.1007/978-3-032-37685-5_41"><img src="https://img.shields.io/badge/paper-ECML--PKDD%202026-8250df" alt="Paper: ECML-PKDD 2026"></a>
  <a href="https://pypi.org/project/tsseg/"><img src="https://img.shields.io/pypi/v/tsseg?logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python 3.10 | 3.11 | 3.12"></a>
  <a href="https://github.com/fchavelli/tsseg/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/fchavelli/tsseg/ci.yml?branch=main&label=CI&logo=github" alt="CI status"></a>
  <a href="https://fchavelli.github.io/tsseg/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=readthedocs&logoColor=white" alt="Documentation"></a>
  <a href="https://huggingface.co/spaces/fchavelli/tsseg"><img src="https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-ffcc4d?logo=huggingface&logoColor=white" alt="Interactive demo on Hugging Face Spaces"></a>
  <a href="https://github.com/fchavelli/tsseg/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

<p align="center">
  <a href="https://fchavelli.github.io/tsseg/"><b>Documentation</b></a> ·
  <a href="https://huggingface.co/spaces/fchavelli/tsseg"><b>Live demo</b></a> ·
  <a href="https://fchavelli.github.io/tsseg/detectors/index.html"><b>Detectors</b></a> ·
  <a href="https://fchavelli.github.io/tsseg/evaluation.html"><b>Metrics</b></a> ·
  <a href="https://fchavelli.github.io/tsseg/changelog.html"><b>Changelog</b></a> ·
  <a href="https://github.com/fchavelli/TSB-SEG"><b>Benchmark</b></a>
</p>

<p align="center">
  <a href="https://doi.org/10.1007/978-3-032-37685-5_41"><img src="https://raw.githubusercontent.com/fchavelli/tsseg/main/assets/ecml-pkdd-2026-banner.svg" alt="Published at ECML-PKDD 2026 (Demo Track, Naples): tsseg: An Interactive Toolkit for Time Series Segmentation, Lecture Notes in Computer Science, Springer, doi:10.1007/978-3-032-37685-5_41" width="820"></a>
</p>

## What is tsseg?

**tsseg** gathers classic and recent algorithms for **time series segmentation**
behind a single API, with the metrics to evaluate them and an interactive demo to
compare them. It covers the two faces of the problem:

| | Change point detection | State detection |
| --- | --- | --- |
| **Question** | _Where_ does the behaviour of the series change? | _Which_ recurring state is each time point in? |
| **Output** | the sorted indices where a new segment starts | one integer label per time point |
| **Example** | `[580, 1125, 2050, 2465, 3995]` | `[0, 0, …, 2, 2, …, 1, 1, …, 0, 0, …]` |

<p align="center">
  <img src="https://raw.githubusercontent.com/fchavelli/tsseg/main/assets/demo.png" alt="A MoCap recording segmented in the tsseg demo: ground truth, change point detection and state detection" width="720">
  <br>
  <sub>A motion-capture recording bundled with tsseg, as shown in the interactive demo: ground truth (top),
  the change points of a change point detector (middle) and the states of a state detector (bottom).</sub>
</p>

## Highlights

- **One API for every method.** Each detector is a scikit-learn estimator that
  follows the segmenter interface of [aeon](https://github.com/aeon-toolkit/aeon):
  `fit`, `predict`, `fit_predict`, `get_params`, `clone`.
- **Unsupervised or guided.** Run a detector without knowing how many segments
  to expect, or pass the number of change points or states to its constructor
  when you know it. Tags tell which modes each detector supports.
- **Metrics for both tasks.** F1, covering, Hausdorff distance and Gaussian F1
  for change points; ARI, AMI, NMI, their time sensitive counterparts, and the interpretable SMS for states.
- **Light core, opt-in extras.** The core needs NumPy, pandas, SciPy,
  scikit-learn and statsmodels. PyTorch, TensorFlow, Prophet or numba are
  installed only for the detectors that use them.
- **Fast classics.** PELT, BinSeg, BottomUp, DynP and Window run on a vendored
  copy of [ruptures](https://github.com/deepcharles/ruptures) with an optional
  numba backend.
- **Benchmarked.** [TSB-SEG](https://github.com/fchavelli/TSB-SEG) (NeurIPS
  2026) evaluates 26 tsseg detectors on 592 series from 8 datasets, for both
  tasks, and publishes every run.
- **Try it in the browser.** The
  [interactive demo](https://huggingface.co/spaces/fchavelli/tsseg) loads a custom
  series, runs the detectors and compares their results, with nothing to
  install.

## Installation

```bash
pip install tsseg
```

Most detectors work with this alone. Those built on heavier libraries need an
extra, for instance `pip install "tsseg[torch]"`, or all of them at once with
`pip install "tsseg[all]"`:

| Extra | Adds | For |
| --- | --- | --- |
| `tsseg[accelerators]` | Numba, Cython | AutoPlait, CLaP, ClaSP and E-Agglo (required); HDP-HSMM (speed-up) |
| `tsseg[aeon]` | aeon's time series classifiers | `ClapDetector` |
| `tsseg[torch]` | PyTorch | `TireDetector`, `Time2StateDetector`, `E2USDDetector` |
| `tsseg[tglad]` | PyTorch, NetworkX | `TGLADDetector` |
| `tsseg[tscp2]` | TensorFlow, TCN layer | `TSCP2Detector` |
| `tsseg[snlds]` | TensorFlow, TensorFlow Probability | `SNLDSDetector` |
| `tsseg[prophet]` | Prophet | `ProphetDetector` |
| `tsseg[beast]` | Rbeast | `BeastDetector` |
| `tsseg[changeforest]` | changeforest | `ChangeForestDetector` |
| `tsseg[fluss]` | stumpy | `FLUSSDetector` |
| `tsseg[docs]` | Sphinx toolchain | building the documentation |
| `tsseg[all]` | everything above | |

The `tsseg[patss]` extra installs nothing: the deprecated `PatssDetector` needs
`npbad`, which is not on PyPI and must be installed by hand.

`tsseg[tscp2]`, hence `tsseg[all]`, needs Python 3.10 to 3.12: the TensorFlow
versions it pins (below 2.19) have no wheel for Python 3.13 or later. On Linux,
`torch` from PyPI comes with the CUDA libraries; without a GPU, install its CPU
build first, as the CI does:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

<details>
<summary><b>From source</b></summary>

In a virtual environment:

```bash
git clone https://github.com/fchavelli/tsseg.git
cd tsseg
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
```

`make install` does the same in `.venv` (`make help` lists its options, such as
`EXTRAS=dev,torch`, `VENV_PYTHON=python3.12` or
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`).

With conda, `make install-conda` creates the `tsseg-env` environment from
`environment.yml` and installs tsseg in it (it needs `conda` in your `PATH`, or
set in a `.env` file, see `.env.example`):

```bash
make install-conda
conda activate tsseg-env
```

</details>

<details>
<summary><b>Without TensorFlow</b></summary>

Only `TSCP2Detector` (`tsseg[tscp2]`) and `SNLDSDetector` (`tsseg[snlds]`) need
TensorFlow. If it causes trouble on your platform, skip `[all]` and pick the
extras you need:

```bash
pip install -e ".[torch,prophet,aeon,accelerators]"
```

</details>

## Quick start

Segment a motion-capture recording, then score both results against its ground
truth (the state detector needs `pip install "tsseg[accelerators,aeon]"`):

```python
from tsseg import load_mocap
from tsseg.algorithms import BottomUpDetector, ClapDetector
from tsseg.algorithms.utils import extract_cps
from tsseg.metrics import Covering, StateMatchingScore

# 4 channels (arms and legs), one activity label per time point
X, y = load_mocap(trial=0)  # X: (4579, 4), y: (4579,)
true_cps = extract_cps(y)  # [ 588 1200 2006 2530 3282 4048]

# Change point detection: where does the movement change?
cps = BottomUpDetector(model="rbf", pen_scale="bic").fit_predict(X)
print(cps)  # [ 580 1125 2050 2465 3995]
print(Covering().compute(true_cps, cps, n_timepoints=len(X))["score"])  # 0.748

# State detection: which activity is each time point in?
states = ClapDetector().fit_predict(X)
print(StateMatchingScore().compute(y, states)["score"])  # 0.700
```

Know how many segments to expect? Pass it to the constructor (guided mode):

```python
from tsseg.algorithms import DynpDetector

print(DynpDetector(model="rbf", n_cps=6).fit_predict(X))
# [ 595 1145 2160 2455 3300 4115]
```

Detectors accept NumPy arrays of shape `(n_timepoints, n_channels)` (or 1-D for
univariate series), pandas Series and DataFrames. The
[getting started guide](https://fchavelli.github.io/tsseg/guides/getting-started.html)
goes further.

## Algorithms

**Unsup.**: runs without being told the number of change points or states.
**Guided**: takes that number in its constructor (`n_cps`, `n_states`, …).
📦: needs an optional dependency (see [Installation](https://github.com/fchavelli/tsseg#installation)).
Each name links to the documentation page of the detector.

<details open>
<summary><b>Change point detection (22)</b></summary>

| Detector | Approach | Unsup. | Guided | Reference |
| --- | --- | :---: | :---: | --- |
| [AMOC](https://fchavelli.github.io/tsseg/detectors/amoc.html)<br>`AmocDetector` | single best split | ✓ | | Killick et al. (2012) |
| [BEAST](https://fchavelli.github.io/tsseg/detectors/beast.html) 📦<br>`BeastDetector` | Bayesian model averaging of trend and seasonality | ✓ | ✓ | Zhao et al. (2019), RSE |
| [BinSeg](https://fchavelli.github.io/tsseg/detectors/binseg.html)<br>`BinSegDetector` | greedy top-down splitting | ✓ | ✓ | Bai (1997); Fryzlewicz (2014) |
| [BOCD](https://fchavelli.github.io/tsseg/detectors/bocd.html)<br>`BOCDDetector` | Bayesian run-length posterior | ✓ | ✓ | Adams & MacKay (2007) |
| [BottomUp](https://fchavelli.github.io/tsseg/detectors/bottomup.html)<br>`BottomUpDetector` | greedy bottom-up merging | ✓ | ✓ | Keogh et al. (2001), ICDM |
| [changeforest](https://fchavelli.github.io/tsseg/detectors/changeforest.html) 📦<br>`ChangeForestDetector` | random forest classifier between segments | ✓ | ✓ | Londschien et al. (2023), JMLR |
| [ChangeFinder](https://fchavelli.github.io/tsseg/detectors/changefinder.html)<br>`ChangeFinderDetector` | two-stage autoregressive outlier scores | ✓ | ✓ | Takeuchi & Yamanishi (2006), TKDE |
| [ClaSP](https://fchavelli.github.io/tsseg/detectors/clap.html) 📦<br>`ClaspDetector` | classification score profile | ✓ | ✓ | Ermshaus et al. (2023), DMKD |
| [DynP](https://fchavelli.github.io/tsseg/detectors/dynp.html)<br>`DynpDetector` | exact dynamic programming | | ✓ | Auger & Lawrence (1989) |
| [E-Agglo](https://fchavelli.github.io/tsseg/detectors/eagglo.html) 📦<br>`EAggloDetector` | agglomerative, energy distance | ✓ | | Matteson & James (2014), JASA |
| [ESPRESSO](https://fchavelli.github.io/tsseg/detectors/espresso.html)<br>`EspressoDetector` | entropy and shape of the matrix profile | | ✓ | Deldari et al. (2020), UbiComp |
| [FLUSS](https://fchavelli.github.io/tsseg/detectors/fluss.html) 📦<br>`FLUSSDetector` | arc curve of the matrix profile | ✓ | ✓ | Gharghabi et al. (2017), ICDM |
| [GGS](https://fchavelli.github.io/tsseg/detectors/ggs.html)<br>`GreedyGaussianDetector` | greedy Gaussian likelihood | ✓ | ✓ | Hallac et al. (2019), ADAC |
| [iCID](https://fchavelli.github.io/tsseg/detectors/icid.html)<br>`ICIDDetector` | isolation distributional kernel | ✓ | | Cao et al. (2024), JAIR |
| [IGTS](https://fchavelli.github.io/tsseg/detectors/igts.html)<br>`InformationGainDetector` | information gain, top-down | ✓ | ✓ | Sadri et al. (2017), PMC |
| [PELT](https://fchavelli.github.io/tsseg/detectors/pelt.html)<br>`PeltDetector` | exact penalised search with pruning | ✓ | | Killick et al. (2012), JASA |
| [Prophet](https://fchavelli.github.io/tsseg/detectors/prophet.html) 📦<br>`ProphetDetector` | trend change points of Prophet | ✓ | ✓ | Taylor & Letham (2018) |
| [RuLSIF](https://fchavelli.github.io/tsseg/detectors/rulsif.html)<br>`RuLSIFDetector` | relative density-ratio estimation | ✓ | ✓ | Liu et al. (2013), Neural Netw. |
| [tGLAD](https://fchavelli.github.io/tsseg/detectors/tglad.html) 📦<br>`TGLADDetector` | changes of conditional independence graphs | ✓ | | Imani & Shrivastava (2023) |
| [TIRE](https://fchavelli.github.io/tsseg/detectors/tire.html) 📦<br>`TireDetector` | autoencoder, time-invariant features | ✓ | ✓ | De Ryck et al. (2021), IEEE TSP |
| [TS-CP2](https://fchavelli.github.io/tsseg/detectors/tscp2.html) 📦<br>`TSCP2Detector` | contrastive predictive coding | ✓ | ✓ | Deldari et al. (2021), WWW |
| [Window](https://fchavelli.github.io/tsseg/detectors/window.html)<br>`WindowDetector` | sliding-window discrepancy | ✓ | ✓ | Basseville & Nikiforov (1993) |

</details>

<details open>
<summary><b>State detection (9)</b></summary>

| Detector | Approach | Unsup. | Guided | Reference |
| --- | --- | :---: | :---: | --- |
| [AutoPlait](https://fchavelli.github.io/tsseg/detectors/autoplait.html) 📦<br>`AutoPlaitDetector` | MDL-based regime mining | ✓ | | Matsubara et al. (2014), SIGMOD |
| [CLaP](https://fchavelli.github.io/tsseg/detectors/clap.html) 📦<br>`ClapDetector` | classification label profile | ✓ | ✓ | Ermshaus et al. (2024) |
| [E2USD](https://fchavelli.github.io/tsseg/detectors/e2usd.html) 📦<br>`E2USDDetector` | dual-view embeddings, clustering | ✓ | ✓ | Lai et al. (2024), WWW |
| [Gaussian HMM](https://fchavelli.github.io/tsseg/detectors/gaussian_hmm.html)<br>`GaussianHMMDetector` | hidden Markov model fitted by EM | ✓ | ✓ | Rabiner (1989) |
| [HDP-HSMM](https://fchavelli.github.io/tsseg/detectors/hdp_hsmm.html)<br>`HdpHsmmDetector` | Bayesian nonparametric semi-Markov model | ✓ | ✓ | Johnson & Willsky (2013), JMLR |
| [Hidalgo](https://fchavelli.github.io/tsseg/detectors/hidalgo.html)<br>`HidalgoDetector` | local intrinsic dimension | | ✓ | Allegra et al. (2020), Sci. Rep. |
| [SNLDS](https://fchavelli.github.io/tsseg/detectors/snlds.html) 📦<br>`SNLDSDetector` | switching nonlinear dynamical system (experimental) | | ✓ | Dong et al. (2020), ICML |
| [TICC](https://fchavelli.github.io/tsseg/detectors/ticc.html)<br>`TiccDetector` | Toeplitz inverse covariance clustering | ✓ | ✓ | Hallac et al. (2017), KDD |
| [Time2State](https://fchavelli.github.io/tsseg/detectors/time2state.html) 📦<br>`Time2StateDetector` | contrastive encoder, clustering | ✓ | ✓ | Wang et al. (2023), SIGMOD |

</details>

Also available: [`RandomDetector`](https://fchavelli.github.io/tsseg/detectors/random.html),
a random baseline for both tasks.

## Evaluation metrics

Every metric returns a dictionary with at least a `"score"` key:
`metric.compute(y_true, y_pred)["score"]`.

**Change point detection**, on sorted change point indices:

| Metric | What it measures |
| --- | --- |
| `F1Score` | precision and recall of the detected change points, within a margin |
| `GaussianF1Score` | F1 where each detection earns a reward that decays with its distance to the true change point |
| `Covering` | how well the predicted segments cover the true ones |
| `BidirectionalCovering` | covering in both directions, true to predicted and predicted to true |
| `HausdorffDistance` | the largest distance from a change point to the nearest one of the other set |

> **Tip:** these metrics need the length of the series: pass
> `compute(y_true, y_pred, n_timepoints=len(X))`.

**State detection**, on one label per time point:

| Metric | What it measures |
| --- | --- |
| `AdjustedRandIndex` (ARI), `AdjustedMutualInformation` (AMI), `NormalizedMutualInformation` (NMI) | agreement between the two labellings, as clusterings |
| `WeightedAdjustedRandIndex` (WARI), `WeightedNormalizedMutualInformation` (WNMI) | ARI and NMI with each time point weighted by its distance to the nearest true boundary |
| `StateMatchingScore` (SMS) | matches predicted to true states, then scores four types of errors |

WARI, WNMI and SMS come from
[Toward Interpretable Evaluation Measures for Time Series Segmentation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/09425891e393e64b0535194a81ba15b7-Abstract-Conference.html)
(NeurIPS 2025).

## Interactive demo

<a href="https://huggingface.co/spaces/fchavelli/tsseg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg" alt="Open the demo in Hugging Face Spaces"></a>

The [tsseg demo](https://huggingface.co/spaces/fchavelli/tsseg) on Hugging Face
Spaces runs in the browser: load a series, run segmentation algorithms, then
compare and evaluate their results, with nothing to install.

## Contributing

Bug reports, questions and new detectors are welcome in the
[issues](https://github.com/fchavelli/tsseg/issues). From a clone:

```bash
make test                                # run the test suite, as in CI (long)
make test PYTEST_ARGS="-k PeltDetector"  # run a selection of tests
make lint                                # ruff check + ruff format --check, as in CI
make docs                                # build the documentation, as in CI (needs tsseg[docs])
```

The [contributing guide](https://fchavelli.github.io/tsseg/guides/contributing.html)
covers the development setup, the code style and the tests a new detector must
pass.

## Citation

If tsseg helps your work, please cite the
[ECML-PKDD 2026 paper](https://doi.org/10.1007/978-3-032-37685-5_41) (also
available from GitHub's *Cite this repository* button, through
[`CITATION.cff`](https://github.com/fchavelli/tsseg/blob/main/CITATION.cff)):

```bibtex
@inproceedings{tsseg2026,
  title     = {tsseg: An Interactive Toolkit for Time Series Segmentation},
  author    = {Chavelli, F{\'e}lix and Ermshaus, Arik and Yang, Fan and Sch{\"a}fer, Patrick and Paparrizos, John and Boniol, Paul},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track, Demo Track and Industrial Track},
  series    = {Lecture Notes in Computer Science},
  year      = {2026},
  pages     = {391--394},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-37685-5_41}
}
```

If you compare segmentation methods, or use the results of the
[TSB-SEG benchmark](https://github.com/fchavelli/TSB-SEG), please also cite:

```bibtex
@inproceedings{chavelli2026tsbseg,
  title     = {{TSB-SEG}: A Systematic Time-Series Segmentation Benchmark},
  author    = {Chavelli, F{\'e}lix and Ermshaus, Arik and Yang, Fan and Sch{\"a}fer, Patrick and Paparrizos, John and Boniol, Paul},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

If you use WARI, WNMI or SMS, please also cite:

```bibtex
@inproceedings{chavelli2025interpretable,
  title     = {Toward Interpretable Evaluation Measures for Time Series Segmentation},
  author    = {Chavelli, F{\'e}lix and Boniol, Paul and Thomazo, Micha{\"e}l},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

Each detector also cites its original paper on its
[documentation page](https://fchavelli.github.io/tsseg/detectors/index.html).

## Contributors

- Félix Chavelli (Inria, ENS, Paris)
- Arik Ermshaus (Humboldt-Universität, Berlin)
- Fan Yang (The Ohio State University, Columbus)
- Paul Boniol (Inria, ENS, Paris)
- Patrick Schäfer (Humboldt-Universität, Berlin)
- John Paparrizos (The Ohio State University, AUTh, Columbus)

## License

tsseg is released under the [AGPL-3.0](https://github.com/fchavelli/tsseg/blob/main/LICENSE).

### Third-party components

Several algorithms bundle adapted or vendored code under their own licenses.

<details>
<summary>Components, licenses and sources</summary>

| Component | License | Source |
|---|---|---|
| `aeon` (base, EAgglo, Hidalgo, HMM, IGTS) | BSD-3 | [aeon-toolkit/aeon](https://github.com/aeon-toolkit/aeon) |
| `ruptures/` (vendored v1.1.8) | BSD-2 | [deepcharles/ruptures](https://github.com/deepcharles/ruptures) |
| `bocd/` | MIT | [hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection) |
| `beast/` (wraps Rbeast, not bundled) | GPL-2.0-or-later | [zhaokg/Rbeast](https://github.com/zhaokg/Rbeast) |
| `clap/` (ClaSP / CLaP) | BSD-3 | [ermshaua/claspy](https://github.com/ermshaua/claspy) |
| `ggs/` | BSD-2 | [cvxgrp/GGS](https://github.com/cvxgrp/GGS) |
| `icid/` | GPLv3 | [IsolationKernel/iCID](https://github.com/IsolationKernel/iCID) |
| `patss/` | MIT | [KU Leuven DTAI](https://gitlab.kuleuven.be/u0143709/patss) |
| `rulsif/` (port of the authors' MATLAB code) | MIT | [anewgithubname/change_detection](https://github.com/anewgithubname/change_detection) |
| `snlds/vendor/` | Apache-2.0 | [google-research/snlds](https://github.com/google-research/google-research/tree/master/snlds) |
| `tglad/vendor/` (uGLAD) | Non-Commercial | [Harshs27/tGLAD](https://github.com/Harshs27/tGLAD) |
| `autoplait/` (translation of the authors' C code) | not specified | [Matsubara et al.](https://www.dm.sanken.osaka-u.ac.jp/~yasuko/software.html) |
| `e2usd/` | not specified | [AI4CTS/E2Usd](https://github.com/AI4CTS/E2Usd) |
| `espresso/` | not specified | [cruiseresearchgroup/ESPRESSO](https://github.com/cruiseresearchgroup/ESPRESSO) |
| `ticc/` | BSD-2 | [davidhallac/TICC](https://github.com/davidhallac/TICC) |
| `time2state/` | MIT | [Lab-ANT/Time2State](https://github.com/Lab-ANT/Time2State) |
| `tire/` | not specified | [De Ryck, De Vos & Bertrand (KU Leuven)](https://github.com/deryckt/TIRE) |
| `tscp2/` | not specified | [Cruise Research Group](https://github.com/cruiseresearchgroup/TSCP2) |

Each directory above that bundles code under a published license contains a
copy of it (`LICENSE`, or `LICENCE` in `patss/`). The aeon license is copied in
`eagglo/`, `hidalgo/`, `hmm/` and `igts/`, and also covers the base classes in
`tsseg/algorithms/base.py`. The sources marked *not specified* publish no
license. Rbeast, which `beast/` only wraps, is installed by `tsseg[beast]`;
tsseg follows the license of its R package, GPL (>= 2), although its PyPI
wheel declares MIT.

</details>
