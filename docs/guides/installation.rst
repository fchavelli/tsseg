Installation
============

From PyPI:

.. code-block:: bash

   pip install tsseg

Most detectors work with this alone. Those built on heavier libraries need an
extra, for instance ``pip install "tsseg[torch]"``, or all of them at once with
``pip install "tsseg[all]"``.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Extra
     - Adds
     - For
   * - ``tsseg[accelerators]``
     - Numba, Cython
     - AutoPlait, CLaP, ClaSP and E-Agglo (required); HDP-HSMM (speed-up)
   * - ``tsseg[aeon]``
     - aeon's time series classifiers
     - ``ClapDetector``
   * - ``tsseg[torch]``
     - PyTorch
     - ``TireDetector``, ``Time2StateDetector``, ``E2USDDetector``
   * - ``tsseg[tglad]``
     - PyTorch, NetworkX
     - ``TGLADDetector``
   * - ``tsseg[tscp2]``
     - TensorFlow, TCN layer
     - ``TSCP2Detector``
   * - ``tsseg[snlds]``
     - TensorFlow, TensorFlow Probability
     - ``SNLDSDetector``
   * - ``tsseg[prophet]``
     - Prophet
     - ``ProphetDetector``
   * - ``tsseg[beast]``
     - Rbeast
     - ``BeastDetector``
   * - ``tsseg[changeforest]``
     - changeforest
     - ``ChangeForestDetector``
   * - ``tsseg[fluss]``
     - stumpy
     - ``FLUSSDetector``
   * - ``tsseg[docs]``
     - Sphinx toolchain
     - building the documentation
   * - ``tsseg[all]``
     - everything above
     -

The ``tsseg[patss]`` extra installs nothing: the deprecated ``PatssDetector``
needs ``npbad``, which is not on PyPI and must be installed by hand.

On Python 3.14, ``tsseg[all]`` leaves out ``tsseg[tscp2]`` and
``tsseg[snlds]``: TensorFlow has no stable release for it yet, nor tf-keras any
release, so ``TSCP2Detector`` and ``SNLDSDetector`` need Python 3.10 to 3.13 for
now. On Linux, ``torch`` from PyPI comes with the CUDA libraries; without a GPU,
install its CPU build first, as the CI does:
``pip install torch --index-url https://download.pytorch.org/whl/cpu``.

Every detector imports without its extra: ``from tsseg.algorithms import *``
works on a bare ``pip install tsseg``. Constructing a detector whose
dependencies are missing raises ``ModuleNotFoundError`` with the extra to
install, e.g. ``ClapDetector requires numba, aeon: pip install
"tsseg[accelerators,aeon]"``. The modules a detector needs are listed in its
``python_dependencies`` tag (``ClapDetector.get_class_tag("python_dependencies")``),
and the :doc:`taxonomy <../taxonomy>` can filter the detectors that need no
extra.

From source
-----------

In a virtual environment:

.. code-block:: bash

   git clone https://github.com/fchavelli/tsseg.git
   cd tsseg
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -e ".[all,dev]"

``make install`` does the same in ``.venv`` (``make help`` lists its options,
such as ``EXTRAS=dev,torch``, ``VENV_PYTHON=python3.12`` or
``TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu``).

With conda, ``make install-conda`` creates the ``tsseg-env`` environment from
``environment.yml`` and installs tsseg in it (it needs ``conda`` in your
``PATH``, or set in a ``.env`` file, see ``.env.example``):

.. code-block:: bash

   make install-conda
   conda activate tsseg-env

Without TensorFlow
------------------

Only ``TSCP2Detector`` (``tsseg[tscp2]``) and ``SNLDSDetector``
(``tsseg[snlds]``) need TensorFlow. If it causes trouble on your platform, skip
``[all]`` and pick the extras you need:

.. code-block:: bash

   pip install -e ".[torch,prophet,aeon,accelerators]"
