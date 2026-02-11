"""Local base classes for tsseg algorithms.

This module vendors a minimal subset of the aeon estimator stack needed by
tsseg's detectors. It provides replacements for :class:`BaseSegmenter` and its
superclasses while avoiding a runtime dependency on the external ``aeon``
package. The implementation is transplanted from ``aeon`` commit history but
trimmed to the functionality used within this repository.

Notes
-----
Two key simplifications are applied compared to upstream aeon:

* Soft dependency checks are no-ops. The original aeon base classes verify
  additional packages via :func:`aeon.utils.validation._dependencies`. Since
  tsseg does not expose a plugin system, we skip those checks.
* Only array and pandas based inputs are supported. aeon supports a larger
  zoo of container types; here we keep the numpy/pandas conversion pathway that
  is exercised by the detectors.

The public API matches aeon's counterparts closely, so detector subclasses can
inherit without modifications. Any missing method raises a meaningful error to
help diagnose future divergence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

# Public surface of this module.
__all__ = ["BaseAeonEstimator", "BaseSeriesEstimator", "BaseSegmenter"]

# ---------------------------------------------------------------------------
# Type metadata
# ---------------------------------------------------------------------------

# These constants mirror ``aeon.utils.data_types`` but are intentionally minimal.
SERIES_INPUT_TYPES = (pd.Series, pd.DataFrame, np.ndarray)
SERIES_INNER_TYPES = {"np.ndarray", "pd.DataFrame"}


def _infer_inner_type(name: str) -> str:
	"""Return the simplified name of a supported inner type."""

	if name.startswith("numpy") or name == "ndarray":
		return "np.ndarray"
	if name in {"Series", "DataFrame"}:
		return f"pd.{name}"
	return name


# ---------------------------------------------------------------------------
# Base estimator hierarchy
# ---------------------------------------------------------------------------


class BaseAeonEstimator(BaseEstimator, ABC):
	"""Minimal clone/reset/tag functionality matching aeon's BaseAeonEstimator."""

	_tags: Dict[str, Any] = {
		"python_version": None,
		"python_dependencies": None,
		"cant_pickle": False,
		"non_deterministic": False,
		"algorithm_type": None,
		"capability:missing_values": False,
		"capability:multithreading": False,
	}

	def __init__(self) -> None:
		self.is_fitted = False
		self._tags_dynamic: Dict[str, Any] = {}

		super().__init__()

	# ------------------------------------------------------------------
	# Tag utilities
	# ------------------------------------------------------------------

	@classmethod
	def get_class_tags(cls) -> Dict[str, Any]:
		"""Collect class tags respecting inheritance order."""

		collected: Dict[str, Any] = {}
		for parent in reversed(cls.__mro__[:-2]):
			if hasattr(parent, "_tags"):
				collected.update(getattr(parent, "_tags"))
		return deepcopy(collected)

	@classmethod
	def get_class_tag(
		cls,
		tag_name: str,
		raise_error: bool = True,
		tag_value_default: Any | None = None,
	) -> Any:
		"""Return a tag value declared on the class hierarchy."""

		tags = cls.get_class_tags()
		if tag_name in tags:
			return tags[tag_name]
		if raise_error:
			raise ValueError(f"Tag with name {tag_name} could not be found.")
		return tag_value_default

	def get_tags(self) -> Dict[str, Any]:
		"""Return tags with dynamic overrides applied."""

		tags = self.get_class_tags()
		tags.update(self._tags_dynamic)
		return deepcopy(tags)

	def get_tag(
		self,
		tag_name: str,
		raise_error: bool = True,
		tag_value_default: Any | None = None,
	) -> Any:
		"""Return a single tag value, including dynamic overrides."""

		tags = self.get_tags()
		if tag_name in tags:
			return tags[tag_name]
		if raise_error:
			raise ValueError(f"Tag with name {tag_name} could not be found.")
		return tag_value_default

	def set_tags(self, **tag_dict: Any) -> "BaseAeonEstimator":
		"""Set dynamic tags and return ``self`` for chaining."""

		self._tags_dynamic.update(deepcopy(tag_dict))
		return self

	# ------------------------------------------------------------------
	# Fitted parameter helpers
	# ------------------------------------------------------------------

	def get_fitted_params(self, deep: bool = True) -> Dict[str, Any]:
		self._check_is_fitted()
		return self._collect_fitted_params(self, deep)

	def _collect_fitted_params(self, est: Any, deep: bool) -> Dict[str, Any]:
		params: Dict[str, Any] = {}
		keys = [attr for attr in dir(est) if attr.endswith("_") and not attr.startswith("_")]
		for key in keys:
			try:
				value = getattr(est, key)
			except AttributeError:
				continue

			params[key] = value
			if deep and isinstance(value, BaseEstimator):
				nested = self._collect_fitted_params(value, deep)
				params.update({f"{key}__{k}": v for k, v in nested.items()})
		return params

	# ------------------------------------------------------------------
	# Reset/clone utilities
	# ------------------------------------------------------------------

	def reset(self, keep: str | Iterable[str] | None = None) -> "BaseAeonEstimator":
		params = self.get_params(deep=False)
		attrs = [attr for attr in vars(self).keys() if "__" not in attr]

		if keep is None:
			keep_set: set[str] = set()
		elif isinstance(keep, str):
			keep_set = {keep}
		else:
			keep_set = set(keep)

		for attr in attrs:
			if attr not in keep_set:
				delattr(self, attr)

		self.__init__(**params)
		return self

	def clone(self, random_state: int | None = None) -> "BaseAeonEstimator":
		clone_obj = type(self)(**self.get_params(deep=False))
		if random_state is not None and hasattr(clone_obj, "set_random_state"):
			clone_obj.set_random_state(random_state)
		return clone_obj

	# ------------------------------------------------------------------
	# Fitted status utilities
	# ------------------------------------------------------------------

	def _check_is_fitted(self) -> None:
		if not getattr(self, "is_fitted", False):
			raise NotFittedError(
				f"This instance of {self.__class__.__name__} has not been fitted yet;"
				" please call `fit` first."
			)


class BaseSeriesEstimator(BaseAeonEstimator):
	"""Base class for single-series estimators with numpy/pandas support."""

	_tags: Dict[str, Any] = {
		"capability:univariate": True,
		"capability:multivariate": False,
		"X_inner_type": "np.ndarray",
		"capability:missing_values": False,
	}

	def __init__(self, axis: int) -> None:
		if axis not in (0, 1):
			raise ValueError("axis should be 0 or 1")
		self.axis = axis
		self.metadata_: Dict[str, Any] = {}
		super().__init__()

	# ------------------------------------------------------------------
	# Data preparation helpers
	# ------------------------------------------------------------------

	def _preprocess_series(self, X: Any, axis: int, store_metadata: bool) -> Any:
		metadata = self._check_X(X, axis)
		if store_metadata:
			self.metadata_ = metadata
		return self._convert_X(X, axis)

	def _check_X(self, X: Any, axis: int) -> Dict[str, Any]:
		if axis not in (0, 1):
			raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

		if isinstance(X, np.ndarray):
			if not (np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, np.floating)):
				raise ValueError("dtype for np.ndarray must be float or int")
		elif isinstance(X, pd.Series):
			if not pd.api.types.is_numeric_dtype(X):
				raise ValueError("pd.Series dtype must be numeric")
		elif isinstance(X, pd.DataFrame):
			if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
				raise ValueError("pd.DataFrame dtype must be numeric")
		else:
			raise ValueError(
				f"Input type of X should be one of {SERIES_INPUT_TYPES}, saw {type(X)}"
			)

		metadata: Dict[str, Any] = {}

		if getattr(X, "ndim", 1) > 2:
			raise ValueError("X must have at most 2 dimensions for multivariate data")

		if getattr(X, "ndim", 1) == 1:
			metadata["multivariate"] = False
			metadata["n_channels"] = 1
		else:
			channel_idx = 0 if axis == 1 else 1
			metadata["multivariate"] = X.shape[channel_idx] > 1
			metadata["n_channels"] = X.shape[channel_idx]

		if isinstance(X, np.ndarray):
			metadata["missing_values"] = np.isnan(X).any()
		else:
			metadata["missing_values"] = X.isna().any().any() if isinstance(X, pd.DataFrame) else X.isna().any()

		if metadata["missing_values"] and not self.get_tag("capability:missing_values"):
			raise ValueError(
				f"Missing values not supported by {self.__class__.__name__}"
			)
		if metadata["multivariate"] and not self.get_tag("capability:multivariate"):
			raise ValueError(
				f"Multivariate data not supported by {self.__class__.__name__}"
			)
		if not metadata["multivariate"] and not self.get_tag("capability:univariate"):
			raise ValueError(
				f"Univariate data not supported by {self.__class__.__name__}"
			)

		return metadata

	def _convert_X(self, X: Any, axis: int) -> Any:
		if axis not in (0, 1):
			raise ValueError(f"Input axis should be 0 or 1, saw {axis}")

		inner_type_name = self.get_tag("X_inner_type")
		if isinstance(inner_type_name, list):
			inner_type_name = inner_type_name[0]
		target = _infer_inner_type(inner_type_name.split(".")[-1])

		if isinstance(X, pd.Series):
			if target == "np.ndarray":
				X = X.to_numpy()
			else:
				X = X.to_frame()
		elif isinstance(X, pd.DataFrame):
			if target == "np.ndarray":
				X = X.to_numpy()
		elif isinstance(X, np.ndarray):
			if target == "pd.DataFrame":
				X = pd.DataFrame(X)
		else:
			raise ValueError(
				f"Unsupported inner type {target} derived from {inner_type_name}"
			)

		if isinstance(X, np.ndarray):
			if X.ndim == 1:
				X = X[np.newaxis, :] if self.axis == 1 else X[:, np.newaxis]
			elif X.ndim == 2 and self.axis != axis:
				X = X.T
		elif isinstance(X, pd.DataFrame):
			if X.ndim == 2 and self.axis != axis:
				X = X.transpose()

		return X


class BaseSegmenter(BaseSeriesEstimator):
	"""Base class for segmentation algorithms, based on aeon's API."""

	_tags: Dict[str, Any] = {
		"X_inner_type": "np.ndarray",
		"fit_is_empty": True,
		"requires_y": False,
		"returns_dense": True,
		"python_dependencies": None,
	}

	# Removed explicit n_segments parameter for flexibility in subclasses.
	# def __init__(self, axis: int, n_segments: int = 2) -> None:
	# 	self.n_segments = n_segments
	# 	super().__init__(axis=axis)

	def __init__(self, axis: int) -> None:
		super().__init__(axis=axis)

	# Public API -------------------------------------------------------

	def fit(self, X: Any, y: Any | None = None, axis: int | None = None) -> "BaseSegmenter":
		if self.get_tag("fit_is_empty"):
			self.is_fitted = True
			return self

		if self.get_tag("requires_y") and y is None:
			raise ValueError("Tag requires_y is true, but fit called with y=None")

		self.reset()

		if axis is None:
			axis = self.axis

		X_inner = self._preprocess_series(X, axis, store_metadata=True)

		if y is not None:
			self._check_y(y)

		self._fit(X=X_inner, y=y)
		self.is_fitted = True
		return self

	def predict(self, X: Any, axis: int | None = None):
		if not self.get_tag("fit_is_empty"):
			self._check_is_fitted()

		if axis is None:
			axis = self.axis

		X_inner = self._preprocess_series(X, axis, store_metadata=False)
		return self._predict(X_inner)

	def fit_predict(self, X: Any, y: Any | None = None, axis: int | None = None):
		self.fit(X, y, axis=axis)
		return self.predict(X, axis=axis)

	# Hooks for subclasses ---------------------------------------------

	def _fit(self, X: Any, y: Any | None):
		return self

	@abstractmethod
	def _predict(self, X: Any):
		...

	# Helper utilities -------------------------------------------------

	def _check_y(self, y: Any) -> None:
		if isinstance(y, np.ndarray):
			if y.ndim > 1:
				raise ValueError("y input as np.ndarray should be 1D")
			if not (np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.floating)):
				raise ValueError("y input must contain floats or ints")
		elif isinstance(y, pd.Series):
			if not pd.api.types.is_numeric_dtype(y):
				raise ValueError("y input as pd.Series must be numeric")
		elif isinstance(y, pd.DataFrame):
			if y.shape[1] > 2:
				raise ValueError("y input as pd.DataFrame should have a single column series")
			if not all(pd.api.types.is_numeric_dtype(y[col]) for col in y.columns):
				raise ValueError("y input as pd.DataFrame must be numeric")
		else:
			raise ValueError(
				f"Error in input type for y: it should be one of {SERIES_INPUT_TYPES}, saw {type(y)}"
			)

	# Convenience converters -------------------------------------------

	@classmethod
	def to_classification(cls, change_points: list[int], length: int) -> np.ndarray:
		labels = np.zeros(length, dtype=int)
		labels[change_points] = 1
		return labels

	@classmethod
	def to_clusters(cls, change_points: list[int], length: int) -> np.ndarray:
		labels = np.zeros(length, dtype=int)
		for cp in change_points:
			labels[cp:] += 1
		return labels


