"""Tests for the declarative parameter constraint system.

Covers:
  * Individual constraint classes (Interval, StrOptions, Options, etc.)
  * Cross-parameter constraints (MutuallyExclusive, DependsOn, etc.)
  * Data-dependent constraints (DataDependent)
  * Schema resolution via MRO
  * validate_params() on real detector instances
  * get_ui_hints() output structure
  * Consistency: schema keys match __init__ parameters
"""

from __future__ import annotations

import importlib
import importlib.util as _ilu
import inspect
import os
import sys
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Helper: import a module by file path without triggering the heavy
# tsseg.algorithms.__init__ (which eagerly imports all 30+ detectors,
# some requiring optional deps like prophet / tensorflow).
# ---------------------------------------------------------------------------

_ALGO_ROOT = os.path.join(
    os.path.dirname(__file__), os.pardir, "tsseg", "algorithms"
)
_ALGO_ROOT = os.path.normpath(_ALGO_ROOT)


def _import_module_from_file(dotted_name: str, filepath: str):
    """Import *filepath* as *dotted_name* if not already loaded."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = _ilu.spec_from_file_location(dotted_name, filepath)
    mod = _ilu.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_algo_package():
    """Register tsseg.algorithms package shell without executing __init__.py."""
    _pkg_name = "tsseg.algorithms"
    if _pkg_name not in sys.modules:
        _pkg_spec = _ilu.spec_from_file_location(
            _pkg_name,
            os.path.join(_ALGO_ROOT, "__init__.py"),
            submodule_search_locations=[_ALGO_ROOT],
        )
        _pkg_mod = _ilu.module_from_spec(_pkg_spec)
        sys.modules[_pkg_name] = _pkg_mod


_ensure_algo_package()

# Now import the modules we actually need.
_import_module_from_file(
    "tsseg.algorithms.base",
    os.path.join(_ALGO_ROOT, "base.py"),
)
_ps_mod = _import_module_from_file(
    "tsseg.algorithms.param_schema",
    os.path.join(_ALGO_ROOT, "param_schema.py"),
)

Closed = _ps_mod.Closed
ConditionalRequired = _ps_mod.ConditionalRequired
DataDependent = _ps_mod.DataDependent
DependsOn = _ps_mod.DependsOn
HasType = _ps_mod.HasType
Interval = _ps_mod.Interval
MutuallyExclusive = _ps_mod.MutuallyExclusive
Options = _ps_mod.Options
ParamDef = _ps_mod.ParamDef
StrOptions = _ps_mod.StrOptions
get_parameter_schema = _ps_mod.get_parameter_schema
get_ui_hints = _ps_mod.get_ui_hints
validate_params = _ps_mod.validate_params
CROSS_CONSTRAINTS_KEY = _ps_mod.CROSS_CONSTRAINTS_KEY


def _can_import(module_name: str) -> bool:
    """Return True if *module_name* can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _load_detector(subpkg: str, filename: str, classname: str):
    """Import a single detector class without triggering algorithms.__init__."""
    dotted = f"tsseg.algorithms.{subpkg}"
    pkg_dir = os.path.join(_ALGO_ROOT, subpkg)
    # register sub-package
    if dotted not in sys.modules:
        pkg_spec = _ilu.spec_from_file_location(
            dotted,
            os.path.join(pkg_dir, "__init__.py"),
            submodule_search_locations=[pkg_dir],
        )
        pkg_mod = _ilu.module_from_spec(pkg_spec)
        sys.modules[dotted] = pkg_mod
        try:
            pkg_spec.loader.exec_module(pkg_mod)
        except Exception:
            pass
    mod_dotted = f"{dotted}.{filename}"
    mod = _import_module_from_file(
        mod_dotted, os.path.join(pkg_dir, f"{filename}.py")
    )
    return getattr(mod, classname)


# Eagerly load the three core detectors that have no heavy deps.
BinSegDetector = _load_detector("binseg", "detector", "BinSegDetector")
PeltDetector = _load_detector("pelt", "detector", "PeltDetector")
BOCDDetector = _load_detector("bocd", "detector", "BOCDDetector")

# Wave-2 detectors (no heavy deps either).
AmocDetector = _load_detector("amoc", "detector", "AmocDetector")
AutoPlaitDetector = _load_detector("autoplait", "detector", "AutoPlaitDetector")
BottomUpDetector = _load_detector("bottomup", "detector", "BottomUpDetector")
ClaspDetector = _load_detector("clap", "clasp_detector", "ClaspDetector")
DynpDetector = _load_detector("dynp", "detector", "DynpDetector")
EspressoDetector = _load_detector("espresso", "detector", "EspressoDetector")
GreedyGaussianDetector = _load_detector("ggs", "detector", "GreedyGaussianDetector")

# Wave-3 detectors (no heavy deps: igts, kcpd, random, window).
InformationGainDetector = _load_detector("igts", "detector", "InformationGainDetector")
KCPDDetector = _load_detector("kcpd", "detector", "KCPDDetector")
RandomDetector = _load_detector("random", "detector", "RandomDetector")
WindowDetector = _load_detector("window", "detector", "WindowDetector")

# ======================================================================
# Unit tests — Interval
# ======================================================================

class TestInterval:
    def test_closed_both(self):
        c = Interval(int, 1, 10, Closed.BOTH)
        assert c.validate(1, "x") is None
        assert c.validate(10, "x") is None
        assert c.validate(5, "x") is None
        assert c.validate(0, "x") is not None
        assert c.validate(11, "x") is not None

    def test_closed_neither(self):
        c = Interval(float, 0, 1, Closed.NEITHER)
        assert c.validate(0.5, "x") is None
        assert c.validate(0, "x") is not None  # > 0 required
        assert c.validate(1, "x") is not None  # < 1 required

    def test_closed_left(self):
        c = Interval(int, 0, None, Closed.LEFT)
        assert c.validate(0, "x") is None
        assert c.validate(999, "x") is None
        assert c.validate(-1, "x") is not None

    def test_closed_right(self):
        c = Interval(int, None, 100, Closed.RIGHT)
        assert c.validate(-999, "x") is None
        assert c.validate(100, "x") is None
        assert c.validate(101, "x") is not None

    def test_unbounded(self):
        c = Interval(float, None, None)
        assert c.validate(-1e18, "x") is None
        assert c.validate(1e18, "x") is None

    def test_none_passthrough(self):
        c = Interval(int, 1, 10, Closed.BOTH)
        assert c.validate(None, "x") is None  # None handled by Options

    def test_type_check_int(self):
        c = Interval(int, 1, 10, Closed.BOTH)
        assert c.validate(3.5, "x") is not None  # must be int

    def test_str_representation(self):
        c = Interval(int, 1, 10, Closed.BOTH)
        assert str(c) == "[1, 10]"
        c2 = Interval(float, 0, 1, Closed.NEITHER)
        assert str(c2) == "(0, 1)"


# ======================================================================
# Unit tests — StrOptions
# ======================================================================

class TestStrOptions:
    def test_valid(self):
        c = StrOptions({"l1", "l2", "rbf"})
        assert c.validate("l2", "x") is None

    def test_invalid(self):
        c = StrOptions({"l1", "l2"})
        assert c.validate("rbf", "x") is not None

    def test_none_passthrough(self):
        c = StrOptions({"a", "b"})
        assert c.validate(None, "x") is None


# ======================================================================
# Unit tests — Options
# ======================================================================

class TestOptions:
    def test_none_sentinel(self):
        c = Options(int, {None})
        assert c.validate(None, "x") is None
        assert c.validate(5, "x") is None
        assert c.validate("foo", "x") is not None

    def test_multi_type(self):
        c = Options((str, int), set())
        assert c.validate("foo", "x") is None
        assert c.validate(5, "x") is None
        assert c.validate(3.14, "x") is not None


# ======================================================================
# Unit tests — HasType
# ======================================================================

class TestHasType:
    def test_valid(self):
        c = HasType((dict,))
        assert c.validate({"a": 1}, "x") is None
        assert c.validate(None, "x") is None  # None passthrough

    def test_invalid(self):
        c = HasType((dict,))
        assert c.validate([1, 2], "x") is not None


# ======================================================================
# Unit tests — MutuallyExclusive
# ======================================================================

class TestMutuallyExclusive:
    def test_one_set(self):
        c = MutuallyExclusive(["a", "b", "c"], required_count=1)
        assert c.validate({"a": 5, "b": None, "c": None}) is None

    def test_none_set(self):
        c = MutuallyExclusive(["a", "b", "c"], required_count=1)
        assert c.validate({"a": None, "b": None, "c": None}) is not None

    def test_two_set(self):
        c = MutuallyExclusive(["a", "b"], required_count=1)
        assert c.validate({"a": 1, "b": 2}) is not None


# ======================================================================
# Unit tests — DependsOn
# ======================================================================

class TestDependsOn:
    def test_satisfied(self):
        c = DependsOn("stride <= window_size", "stride must be <= window_size")
        assert c.validate({"stride": 1, "window_size": 10}) is None

    def test_violated(self):
        c = DependsOn("stride <= window_size", "stride must be <= window_size")
        assert c.validate({"stride": 20, "window_size": 10}) is not None


# ======================================================================
# Unit tests — ConditionalRequired
# ======================================================================

class TestConditionalRequired:
    def test_condition_met_param_present(self):
        c = ConditionalRequired("n_change_points", "semi_supervised == True")
        assert c.validate({"semi_supervised": True, "n_change_points": 3}) is None

    def test_condition_met_param_missing(self):
        c = ConditionalRequired("n_change_points", "semi_supervised == True")
        assert c.validate({"semi_supervised": True, "n_change_points": None}) is not None

    def test_condition_not_met(self):
        c = ConditionalRequired("n_change_points", "semi_supervised == True")
        assert c.validate({"semi_supervised": False, "n_change_points": None}) is None


# ======================================================================
# Unit tests — DataDependent
# ======================================================================

class TestDataDependent:
    def test_no_ctx(self):
        c = DataDependent("window_size < n_samples")
        assert c.validate({"window_size": 100}, data_ctx=None) is None

    def test_satisfied(self):
        c = DataDependent("window_size < n_samples")
        assert c.validate({"window_size": 100}, data_ctx={"n_samples": 200}) is None

    def test_violated(self):
        c = DataDependent("window_size < n_samples", "window_size too large")
        err = c.validate({"window_size": 300}, data_ctx={"n_samples": 200})
        assert err is not None
        assert "window_size too large" in err

    def test_resolve_bound_simple(self):
        c = DataDependent("window_size < n_samples")
        assert c.resolve_bound("window_size", {"n_samples": 500}) == 500

    def test_resolve_bound_offset(self):
        c = DataDependent("n_cps < n_samples - 1")
        assert c.resolve_bound("n_cps", {"n_samples": 100}) == 99

    def test_resolve_bound_division(self):
        c = DataDependent("n_segments <= n_samples // window_size")
        # This pattern doesn't match param "n_segments" directly (window_size isn't in data)
        # so it returns None
        assert c.resolve_bound("n_segments", {"n_samples": 100}) is None


# ======================================================================
# Unit tests — ParamDef
# ======================================================================

class TestParamDef:
    def test_nullable_none(self):
        p = ParamDef(constraint=Interval(int, 1, 10, Closed.BOTH), nullable=True)
        assert p.validate(None, "x") is None

    def test_non_nullable_none(self):
        p = ParamDef(constraint=Interval(int, 1, 10, Closed.BOTH), nullable=False)
        assert p.validate(None, "x") is not None

    def test_options_sentinel_none(self):
        p = ParamDef(constraint=Options(int, {None}))
        assert p.validate(None, "x") is None

    def test_no_constraint(self):
        p = ParamDef(description="anything goes")
        assert p.validate(42, "x") is None
        assert p.validate("foo", "x") is None

    def test_or_semantics_list(self):
        p = ParamDef(
            constraint=[
                StrOptions({"suss", "fft", "acf"}),
                Interval(int, 4, None, Closed.LEFT),
            ]
        )
        assert p.validate("suss", "x") is None
        assert p.validate(10, "x") is None
        assert p.validate(2, "x") is not None  # too small int, not a valid string
        assert p.validate(3.5, "x") is not None


# ======================================================================
# Schema resolution
# ======================================================================

class TestSchemaResolution:
    """Verify that schema keys are a subset of __init__ parameter names."""

    def test_empty_schema(self):
        BaseSegmenter = sys.modules["tsseg.algorithms.base"].BaseSegmenter
        schema = get_parameter_schema(BaseSegmenter)
        assert schema == {} or all(k == CROSS_CONSTRAINTS_KEY for k in schema)

    def test_binseg_schema(self):
        schema = get_parameter_schema(BinSegDetector)
        assert "n_cps" in schema
        assert "model" in schema
        assert "min_size" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    def test_pelt_schema(self):
        schema = get_parameter_schema(PeltDetector)
        assert "model" in schema
        assert "penalty" in schema

    def test_bocd_schema(self):
        schema = get_parameter_schema(BOCDDetector)
        assert "hazard_lambda" in schema
        assert "kappa" in schema

    def test_clap_schema(self):
        ClapDetector = _load_detector("clap", "clap_detector", "ClapDetector")
        schema = get_parameter_schema(ClapDetector)
        assert "window_size" in schema
        assert "classifier" in schema

    @pytest.mark.skipif(
        not _can_import("torch"), reason="torch not installed"
    )
    def test_tire_schema(self):
        TireDetector = _load_detector("tire", "detector", "TireDetector")
        schema = get_parameter_schema(TireDetector)
        assert "window_size" in schema
        assert "stride" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    @pytest.mark.skipif(
        not _can_import("torch"), reason="torch not installed"
    )
    def test_e2usd_schema(self):
        E2USDDetector = _load_detector("e2usd", "detector", "E2USDDetector")
        schema = get_parameter_schema(E2USDDetector)
        assert "window_size" in schema
        assert "step" in schema

    def test_ticc_schema(self):
        TiccDetector = _load_detector("ticc", "detector", "TiccDetector")
        schema = get_parameter_schema(TiccDetector)
        assert "window_size" in schema
        assert "n_states" in schema

    @pytest.mark.skipif(not _can_import("stumpy"), reason="stumpy not installed")
    def test_fluss_schema(self):
        FLUSSDetector = _load_detector("fluss", "detector", "FLUSSDetector")
        schema = get_parameter_schema(FLUSSDetector)
        assert "window_size" in schema
        assert "n_segments" in schema

    # --- Wave 2 ---

    def test_amoc_schema(self):
        schema = get_parameter_schema(AmocDetector)
        assert "min_size" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    def test_autoplait_schema(self):
        schema = get_parameter_schema(AutoPlaitDetector)
        assert "n_cps" in schema

    def test_bottomup_schema(self):
        schema = get_parameter_schema(BottomUpDetector)
        assert "n_cps" in schema
        assert "model" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    def test_clasp_schema(self):
        schema = get_parameter_schema(ClaspDetector)
        assert "n_segments" in schema
        assert "distance" in schema
        assert "validation" in schema

    def test_dynp_schema(self):
        schema = get_parameter_schema(DynpDetector)
        assert "n_cps" in schema
        assert "model" in schema

    def test_eagglo_schema(self):
        EAggloDetector = _load_detector("eagglo", "detector", "EAggloDetector")
        schema = get_parameter_schema(EAggloDetector)
        assert "alpha" in schema
        assert "penalty" in schema

    def test_espresso_schema(self):
        schema = get_parameter_schema(EspressoDetector)
        assert "window_size" in schema
        assert "chain_len" in schema
        assert "n_segments" in schema

    def test_ggs_schema(self):
        schema = get_parameter_schema(GreedyGaussianDetector)
        assert "k_max" in schema
        assert "lamb" in schema

    def test_hidalgo_schema(self):
        HidalgoDetector = _load_detector("hidalgo", "detector", "HidalgoDetector")
        schema = get_parameter_schema(HidalgoDetector)
        assert "K_states" in schema
        assert "zeta" in schema
        assert "burn_in" in schema

    # --- Wave 3 ---

    def test_igts_schema(self):
        schema = get_parameter_schema(InformationGainDetector)
        assert "k_max" in schema
        assert "step" in schema

    def test_kcpd_schema(self):
        schema = get_parameter_schema(KCPDDetector)
        assert "n_cps" in schema
        assert "kernel" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    @pytest.mark.skipif(not _can_import("prophet"), reason="prophet not installed")
    def test_prophet_schema(self):
        ProphetDetector = _load_detector("prophet", "detector", "ProphetDetector")
        schema = get_parameter_schema(ProphetDetector)
        assert "n_changepoints" in schema
        assert "multivariate_strategy" in schema
        assert "tolerance" in schema

    def test_random_schema(self):
        schema = get_parameter_schema(RandomDetector)
        assert "semi_supervised" in schema
        assert "n_change_points" in schema
        assert "n_states" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    @pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
    def test_tglad_schema(self):
        TGLADDetector = _load_detector("tglad", "detector", "TGLADDetector")
        schema = get_parameter_schema(TGLADDetector)
        assert "window_size" in schema
        assert "threshold" in schema
        assert "glad_iterations" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    @pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
    def test_time2state_schema(self):
        Time2StateDetector = _load_detector("time2state", "detector", "Time2StateDetector")
        schema = get_parameter_schema(Time2StateDetector)
        assert "window_size" in schema
        assert "n_states" in schema
        assert "depth" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    @pytest.mark.skipif(not _can_import("tensorflow"), reason="tensorflow not installed")
    def test_tscp2_schema(self):
        TSCP2Detector = _load_detector("tscp2", "detector", "TSCP2Detector")
        schema = get_parameter_schema(TSCP2Detector)
        assert "window_size" in schema
        assert "loss" in schema
        assert "similarity" in schema
        assert CROSS_CONSTRAINTS_KEY in schema

    def test_window_schema(self):
        schema = get_parameter_schema(WindowDetector)
        assert "width" in schema
        assert "n_cps" in schema
        assert "model" in schema
        assert CROSS_CONSTRAINTS_KEY in schema


# ======================================================================
# validate_params() on real detectors
# ======================================================================

class TestValidateParams:
    """Test validation on real detector instances with valid and invalid params."""

    def test_binseg_valid(self):
        det = BinSegDetector(n_cps=3, model="l2", min_size=2)
        errors = validate_params(det)
        assert errors == []

    def test_binseg_invalid_model(self):
        det = BinSegDetector(n_cps=3, model="invalid", min_size=2)
        errors = validate_params(det)
        assert any("model" in e for e in errors)

    def test_binseg_mutual_exclusion(self):
        det = BinSegDetector(n_cps=3)
        # Manually set penalty to trigger mutual exclusion
        det.penalty = 10.0
        errors = validate_params(det)
        assert any("Exactly 1" in e for e in errors)

    def test_pelt_valid(self):
        det = PeltDetector(model="rbf", penalty=5.0)
        errors = validate_params(det)
        assert errors == []

    def test_pelt_invalid_penalty(self):
        det = PeltDetector(penalty=5.0)
        det.penalty = -1.0  # force invalid
        errors = validate_params(det)
        assert any("penalty" in e for e in errors)

    def test_bocd_valid(self):
        det = BOCDDetector(hazard_lambda=100, kappa=1.0, alpha=1.0, beta=1.0)
        errors = validate_params(det)
        assert errors == []

    def test_bocd_invalid_kappa(self):
        det = BOCDDetector(kappa=1.0)
        det.kappa = 0.0  # must be > 0
        errors = validate_params(det)
        assert any("kappa" in e for e in errors)

    def test_bocd_data_dependent(self):
        det = BOCDDetector(min_distance=500)
        errors = validate_params(det, data_ctx={"n_samples": 100, "n_channels": 1})
        assert any("min_distance" in e for e in errors)

    def test_ticc_valid(self):
        TiccDetector = _load_detector("ticc", "detector", "TiccDetector")
        det = TiccDetector(window_size=5, n_states=3)
        errors = validate_params(det)
        assert errors == []

    @pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
    def test_e2usd_depends_on(self):
        E2USDDetector = _load_detector("e2usd", "detector", "E2USDDetector")
        det = E2USDDetector(window_size=50, step=100)
        errors = validate_params(det)
        assert any("step" in e.lower() for e in errors)

    @pytest.mark.skipif(not _can_import("stumpy"), reason="stumpy not installed")
    def test_fluss_valid(self):
        FLUSSDetector = _load_detector("fluss", "detector", "FLUSSDetector")
        det = FLUSSDetector(window_size=10, n_segments=2)
        errors = validate_params(det)
        assert errors == []

    @pytest.mark.skipif(not _can_import("stumpy"), reason="stumpy not installed")
    def test_fluss_data_dependent(self):
        FLUSSDetector = _load_detector("fluss", "detector", "FLUSSDetector")
        det = FLUSSDetector(window_size=500, n_segments=2)
        errors = validate_params(det, data_ctx={"n_samples": 100, "n_channels": 1})
        assert any("window_size" in e for e in errors)

    # --- Wave 3 ---

    def test_igts_valid(self):
        det = InformationGainDetector(k_max=5, step=3)
        errors = validate_params(det)
        assert errors == []

    def test_igts_invalid_k_max(self):
        det = InformationGainDetector(k_max=5)
        det.k_max = 0
        errors = validate_params(det)
        assert any("k_max" in e for e in errors)

    def test_kcpd_valid(self):
        det = KCPDDetector(n_cps=3, kernel="rbf")
        errors = validate_params(det)
        assert errors == []

    def test_kcpd_invalid_kernel(self):
        det = KCPDDetector(n_cps=3)
        det.kernel = "invalid"
        errors = validate_params(det)
        assert any("kernel" in e for e in errors)

    def test_random_valid(self):
        det = RandomDetector(semi_supervised=True, n_change_points=3, n_states=2)
        errors = validate_params(det)
        assert errors == []

    def test_random_conditional_required(self):
        det = RandomDetector()
        det.semi_supervised = True
        det.n_change_points = None
        errors = validate_params(det)
        assert any("n_change_points" in e for e in errors)

    def test_window_valid(self):
        det = WindowDetector(width=50, n_cps=3)
        errors = validate_params(det)
        assert errors == []

    def test_window_invalid_model(self):
        det = WindowDetector(n_cps=3)
        det.model = "invalid"
        errors = validate_params(det)
        assert any("model" in e for e in errors)

    def test_window_mutual_exclusion(self):
        det = WindowDetector(n_cps=3)
        det.pen = 10.0  # force both to be set
        errors = validate_params(det)
        assert any("Exactly 1" in e for e in errors)


# ======================================================================
# get_ui_hints()
# ======================================================================

class TestUIHints:
    def test_binseg_hints(self):
        hints = get_ui_hints(BinSegDetector)
        assert "model" in hints
        assert "choices" in hints["model"]
        assert "l2" in hints["model"]["choices"]
        assert hints["n_cps"]["nullable"]

    @pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
    def test_tire_hints_with_data(self):
        TireDetector = _load_detector("tire", "detector", "TireDetector")
        hints = get_ui_hints(TireDetector, data_ctx={"n_samples": 200, "n_channels": 1})
        assert hints["window_size"]["min"] == 4
        # max should be capped by n_samples via DataDependent
        assert hints["window_size"]["max"] == 200

    def test_bocd_hints(self):
        hints = get_ui_hints(BOCDDetector)
        assert hints["kappa"]["min"] == 0  # low bound
        assert hints["multivariate_strategy"]["choices"] == ["ensembling", "l2"]

    @pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
    def test_e2usd_groups(self):
        E2USDDetector = _load_detector("e2usd", "detector", "E2USDDetector")
        hints = get_ui_hints(E2USDDetector)
        groups = {h.get("group", "") for h in hints.values()}
        assert "windowing" in groups
        assert "training" in groups
        assert "architecture" in groups

    def test_no_schema_returns_empty(self):
        """A class without _parameter_schema should yield empty hints."""
        class _Bare:
            pass
        hints = get_ui_hints(_Bare)
        assert hints == {}


# ======================================================================
# Consistency: schema keys match __init__ params
# ======================================================================

_PILOT_DETECTORS = []

def _get_pilot_detectors():
    """Lazily load pilot detectors to avoid import errors at collection time."""
    global _PILOT_DETECTORS
    if _PILOT_DETECTORS:
        return _PILOT_DETECTORS
    detectors = [
        BinSegDetector, PeltDetector, BOCDDetector,
        AmocDetector, AutoPlaitDetector, BottomUpDetector,
        ClaspDetector, DynpDetector, EspressoDetector,
        GreedyGaussianDetector,
    ]
    try:
        detectors.append(_load_detector("clap", "clap_detector", "ClapDetector"))
    except Exception:
        pass
    if _can_import("torch"):
        try:
            detectors.append(_load_detector("tire", "detector", "TireDetector"))
        except Exception:
            pass
        try:
            detectors.append(_load_detector("e2usd", "detector", "E2USDDetector"))
        except Exception:
            pass
    try:
        detectors.append(_load_detector("fluss", "detector", "FLUSSDetector"))
    except Exception:
        pass
    try:
        detectors.append(_load_detector("ticc", "detector", "TiccDetector"))
    except Exception:
        pass
    try:
        detectors.append(_load_detector("eagglo", "detector", "EAggloDetector"))
    except Exception:
        pass
    try:
        detectors.append(_load_detector("hidalgo", "detector", "HidalgoDetector"))
    except Exception:
        pass
    # Wave-3 detectors
    detectors.extend([
        InformationGainDetector, KCPDDetector, RandomDetector, WindowDetector,
    ])
    if _can_import("prophet"):
        try:
            detectors.append(_load_detector("prophet", "detector", "ProphetDetector"))
        except Exception:
            pass
    if _can_import("torch"):
        try:
            detectors.append(_load_detector("tglad", "detector", "TGLADDetector"))
        except Exception:
            pass
        try:
            detectors.append(_load_detector("time2state", "detector", "Time2StateDetector"))
        except Exception:
            pass
    if _can_import("tensorflow"):
        try:
            detectors.append(_load_detector("tscp2", "detector", "TSCP2Detector"))
        except Exception:
            pass
    _PILOT_DETECTORS = detectors
    return _PILOT_DETECTORS


class TestSchemaConsistency:
    """Verify that schema keys are a subset of __init__ parameter names."""

    def test_schema_keys_match_init(self):
        for cls in _get_pilot_detectors():
            schema = get_parameter_schema(cls)
            sig = inspect.signature(cls.__init__)
            init_params = {
                p.name for p in sig.parameters.values()
                if p.name != "self"
            }
            schema_params = {k for k in schema if k != CROSS_CONSTRAINTS_KEY}
            extra = schema_params - init_params
            assert extra == set(), (
                f"{cls.__name__}: schema declares params not in __init__: {extra}"
            )

    def test_get_test_params_passes_validation(self):
        """For each pilot, _get_test_params() should produce a valid instance."""
        for cls in _get_pilot_detectors():
            if not hasattr(cls, "_get_test_params"):
                continue
            test_params = cls._get_test_params()
            if isinstance(test_params, list):
                test_params = test_params[0]
            try:
                instance = cls(**test_params)
            except Exception:
                continue  # some test params may need deps
            errors = validate_params(instance)
            assert errors == [], (
                f"{cls.__name__}._get_test_params() fails validation: {errors}"
            )


# ======================================================================
# Integration: validation triggered in fit()
# ======================================================================

class TestFitValidation:
    """Verify that BaseSegmenter.fit() triggers param validation."""

    def test_pelt_fit_valid(self):
        det = PeltDetector(model="l2", penalty=10.0, min_size=2)
        X = np.random.randn(200, 1)
        # Should not raise
        det.fit(X, axis=0)

    def test_pelt_fit_data_dependent_violation(self):
        det = PeltDetector(model="l2", penalty=10.0, min_size=50)
        X = np.random.randn(20, 1)  # only 20 samples, min_size*2=100 > 20
        with pytest.raises(ValueError, match="parameter validation failed"):
            det.fit(X, axis=0)

    def test_bocd_predict_validates(self):
        """BOCD has fit_is_empty=True, so validation runs in predict."""
        det = BOCDDetector(min_distance=500)
        X = np.random.randn(100, 1)
        with pytest.raises(ValueError, match="parameter validation failed"):
            det.predict(X, axis=0)
