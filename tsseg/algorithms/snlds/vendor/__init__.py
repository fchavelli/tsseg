"""Vendored CAVI-SNLDS code (Dong et al., ICML 2020), Apache License 2.0.

Copied from Google Research's ``snlds`` package
(https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/snlds):
``forward_backward_algo``, ``model_base``, ``model_cavi_snlds``, ``utils`` and
``examples/config_utils`` (here ``config_utils``). Changes, also noted at the top of each
file: relative imports, and a Keras 3 compatible initial state and call of the
posterior RNN in ``model_cavi_snlds``. The licence is in ``LICENSE`` in this directory.
Importing these modules imports TensorFlow (and TensorFlow Probability).
"""
