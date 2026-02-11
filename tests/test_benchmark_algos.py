"""
Batterie de tests — algorithmes décevants du benchmark tsseg.
Utilise des signaux de difficulté croissante.
"""
import time
import traceback
import warnings
import sys
import numpy as np

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2, suppress=True)

# ── Synthetic signals ───────────────────────────────────────────────
rng = np.random.default_rng(42)
N = 200

# Signal EASY: mean-shift bien séparé (SNR~10)
easy_1d = np.concatenate([
    rng.normal(0, 0.5, N), rng.normal(5, 0.5, N), rng.normal(-3, 0.5, N),
]).reshape(-1, 1)

# Signal MEDIUM: mean-shift plus faible (SNR~2) + variance change
med_1d = np.concatenate([
    rng.normal(0, 1.0, N), rng.normal(2, 0.5, N), rng.normal(-1, 1.5, N),
]).reshape(-1, 1)

# Signal HARD: variance-only change (pas de mean-shift!)
hard_1d = np.concatenate([
    rng.normal(0, 0.3, N), rng.normal(0, 2.0, N), rng.normal(0, 0.3, N),
]).reshape(-1, 1)

# Signal MULTI: 3 channels, mean-shift modéré
multi = np.column_stack([
    np.concatenate([rng.normal(0, 1, N), rng.normal(2, 1, N), rng.normal(-1, 1, N)]),
    np.concatenate([rng.normal(1, 1, N), rng.normal(-1, 1, N), rng.normal(3, 1, N)]),
    np.concatenate([rng.normal(-1, 1, N), rng.normal(1, 1, N), rng.normal(-2, 1, N)]),
])

TRUE_CPS = [200, 400]
SIGNALS = {
    "easy_1d": easy_1d,
    "med_1d":  med_1d,
    "hard_1d": hard_1d,
    "multi":   multi,
}

# ── Helpers ─────────────────────────────────────────────────────────
def cp_error(detected, true_cps=TRUE_CPS, n_samples=600):
    if len(detected) == 0:
        return float("inf"), []
    detected = np.sort(detected)
    true = np.array(true_cps)
    errors = [int(np.min(np.abs(detected - tcp))) for tcp in true]
    return np.mean(errors), errors

def extract_cps_from_output(raw, tags):
    raw = np.asarray(raw).ravel()
    task = tags.get("detector_type", "change_point_detection")
    returns_dense = tags.get("returns_dense", True)
    if task == "state_detection" or not returns_dense:
        diffs = np.where(raw[:-1] != raw[1:])[0] + 1
        return diffs.tolist()
    return raw.tolist()

def test_one(name, det_cls, kw, signal, timeout_s=120):
    t0 = time.time()
    try:
        det = det_cls(**kw)
        det.fit(signal, axis=0)
        raw = det.predict(signal, axis=0)
        elapsed = time.time() - t0
        cps = extract_cps_from_output(raw, det._tags)
        mae, errs = cp_error(cps)
        return {"cps": cps[:15], "n": len(cps), "mae": mae,
                "errs": errs, "t": elapsed, "ok": True}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "err": str(e)[:100], "t": time.time() - t0}

def banner(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")

# ======================================================================
#  1. DynP — always semi-supervised (needs n_cps)
# ======================================================================
banner("DynP")
from tsseg.algorithms.dynp.detector import DynpDetector
for sig_name, sig in SIGNALS.items():
    r = test_one("DynP", DynpDetector,
                 {"n_cps": 2, "model": "l2", "min_size": 2, "jump": 5}, sig)
    status = f"n={r['n']}, MAE={r['mae']:.1f}, errs={r['errs']}, t={r['t']:.2f}s" if r["ok"] else f"ERR: {r['err']}"
    print(f"  {sig_name:<10s}  {status}")

# ======================================================================
#  2. PELT — penalty sensitivity
# ======================================================================
banner("PELT — penalty sweep")
from tsseg.algorithms.pelt.detector import PeltDetector
for sig_name in ["easy_1d", "med_1d", "hard_1d"]:
    sig = SIGNALS[sig_name]
    print(f"  -- {sig_name} --")
    for pen in [1, 3, 5, 10, 30, 50, 100, 500, 1000]:
        r = test_one("PELT", PeltDetector,
                     {"model": "l2", "min_size": 2, "jump": 5, "penalty": pen}, sig)
        if r["ok"]:
            print(f"    pen={pen:>5d}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}")

# ======================================================================
#  3. BinSeg (unguided) — penalty sensitivity
# ======================================================================
banner("BinSeg — unguided penalty sweep")
from tsseg.algorithms.binseg.detector import BinSegDetector
for sig_name in ["easy_1d", "med_1d", "hard_1d"]:
    sig = SIGNALS[sig_name]
    print(f"  -- {sig_name} --")
    for pen in [1, 3, 5, 10, 30, 50, 100, 500, 1000]:
        r = test_one("BinSeg", BinSegDetector,
                     {"n_cps": None, "model": "l2", "min_size": 2, "jump": 5,
                      "penalty": pen}, sig)
        if r["ok"]:
            print(f"    pen={pen:>5d}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}")

# ======================================================================
#  4. BottomUp (unguided) — penalty sensitivity
# ======================================================================
banner("BottomUp — unguided penalty sweep")
from tsseg.algorithms.bottomup.detector import BottomUpDetector
for sig_name in ["easy_1d", "med_1d", "hard_1d"]:
    sig = SIGNALS[sig_name]
    print(f"  -- {sig_name} --")
    for pen in [1, 3, 5, 10, 30, 50, 100, 500, 1000]:
        r = test_one("BottomUp", BottomUpDetector,
                     {"n_cps": None, "model": "l2", "min_size": 2, "jump": 5,
                      "penalty": pen}, sig)
        if r["ok"]:
            print(f"    pen={pen:>5d}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}")

# ======================================================================
#  5. KCPD — kernel + penalty
# ======================================================================
banner("KCPD — kernel x penalty")
from tsseg.algorithms.kcpd.detector import KCPDDetector
for sig_name in ["easy_1d", "med_1d", "hard_1d"]:
    sig = SIGNALS[sig_name]
    print(f"  -- {sig_name} --")
    for kernel in ["linear", "rbf"]:
        for pen in [1, 10, 100, 500]:
            r = test_one("KCPD", KCPDDetector,
                         {"n_cps": None, "pen": pen, "kernel": kernel,
                          "min_size": 2, "jump": 5}, sig)
            if r["ok"]:
                print(f"    {kernel:>7s} pen={pen:>5d}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}")
    # Also guided
    r = test_one("KCPD", KCPDDetector,
                 {"n_cps": 2, "kernel": "linear", "min_size": 2, "jump": 1}, sig)
    if r["ok"]:
        print(f"    GUIDED n_cps=2: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, errs={r['errs']}")

# ======================================================================
#  6. IGTS
# ======================================================================
banner("IGTS")
from tsseg.algorithms.igts.detector import InformationGainDetector
print("  Tags:", InformationGainDetector._tags)
for sig_name in ["multi", "easy_1d"]:
    sig = SIGNALS[sig_name]
    for k_max in [2, 5, 10]:
        for step in [1, 5, 10]:
            r = test_one("IGTS", InformationGainDetector,
                         {"k_max": k_max, "step": step}, sig)
            if r["ok"]:
                print(f"  {sig_name:<10s} k={k_max:>2d} step={step:>2d}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}")
            elif "Univariate" in r.get("err", ""):
                print(f"  {sig_name:<10s} k={k_max:>2d} step={step:>2d}: UNIVARIATE NOT SUPPORTED")
                break  # skip rest for this signal
            else:
                print(f"  {sig_name:<10s} k={k_max:>2d} step={step:>2d}: ERR: {r['err'][:60]}")

# ======================================================================
#  7. TIRE
# ======================================================================
banner("TIRE")
from tsseg.algorithms.tire.detector import TireDetector
base_kw = {
    "window_size": 20, "stride": 1, "domain": "both",
    "intermediate_dim_td": 0, "latent_dim_td": 1,
    "nr_shared_td": 1, "nr_ae_td": 3, "loss_weight_td": 1.0,
    "intermediate_dim_fd": 10, "latent_dim_fd": 1,
    "nr_shared_fd": 1, "nr_ae_fd": 3, "loss_weight_fd": 1.0,
    "nfft": 30, "norm_mode": "timeseries",
    "peak_distance_fraction": 0.01,
    "max_epochs": 30, "patience": 5, "learning_rate": 0.001,
    "axis": 0, "random_state": 0,
}
for sig_name in ["easy_1d", "med_1d", "multi"]:
    sig = SIGNALS[sig_name]
    for n_seg in [None, 3]:
        kw = {**base_kw, "n_segments": n_seg}
        r = test_one("TIRE", TireDetector, kw, sig)
        mode = "guided" if n_seg else "unguided"
        if r["ok"]:
            print(f"  {sig_name:<10s} [{mode:>8s}]: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}, t={r['t']:.1f}s")
        else:
            print(f"  {sig_name:<10s} [{mode:>8s}]: ERR: {r['err'][:80]}, t={r['t']:.1f}s")

# ======================================================================
#  8. HDP-HSMM
# ======================================================================
banner("HDP-HSMM")
from tsseg.algorithms.hdp_hsmm.detector import HdpHsmmDetector
base_kw = {
    "axis": 0, "alpha": 6.0, "gamma": 6.0,
    "init_state_concentration": 6.0,
    "n_iter": 20, "trunc": 100,
    "kappa0": 0.25, "nu0": None,
    "prior_mean": 0.0, "prior_scale": 1.0,
    "dur_alpha": 2.0, "dur_beta": 0.1,
}
for sig_name in ["easy_1d", "med_1d", "multi"]:
    sig = SIGNALS[sig_name]
    for n_max in [20, 3]:
        kw = {**base_kw, "n_max_states": n_max}
        mode = "unguided" if n_max == 20 else "guided"
        r = test_one("HDP-HSMM", HdpHsmmDetector, kw, sig)
        if r["ok"]:
            print(f"  {sig_name:<10s} [{mode:>8s}] n_max={n_max:>2d}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, t={r['t']:.1f}s")
        else:
            print(f"  {sig_name:<10s} [{mode:>8s}] n_max={n_max:>2d}: ERR: {r['err'][:80]}, t={r['t']:.1f}s")

# ======================================================================
#  9. TS-CP2
# ======================================================================
banner("TS-CP2")
try:
    from tsseg.algorithms.tscp2.detector import TSCP2Detector
    base_kw = {
        "window_size": 64, "similarity_threshold": 0.1, "stride": 5,
        "code_size": 32, "nb_filters": 64, "kernel_size": 4,
        "nb_stacks": 2, "dropout_rate": 0.0,
        "batch_size": 64, "epochs": 30,
        "learning_rate": 1e-3, "loss": "nce",
        "temperature": 0.1, "tau": 0.1, "beta": 0.1,
        "similarity": "cosine", "refit_on_predict": False,
        "axis": 0,
    }
    for sig_name in ["easy_1d", "med_1d", "multi"]:
        sig = SIGNALS[sig_name]
        for n_cps in [None, 2]:
            kw = {**base_kw, "n_cps": n_cps}
            mode = "guided" if n_cps else "unguided"
            r = test_one("TS-CP2", TSCP2Detector, kw, sig)
            if r["ok"]:
                print(f"  {sig_name:<10s} [{mode:>8s}]: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}, t={r['t']:.1f}s")
            else:
                print(f"  {sig_name:<10s} [{mode:>8s}]: ERR: {r['err'][:80]}, t={r['t']:.1f}s")
except ImportError as e:
    print(f"  SKIPPED: {e}")

# ======================================================================
# 10. tGLAD
# ======================================================================
banner("tGLAD")
try:
    from tsseg.algorithms.tglad.detector import TGLADDetector
    # tGLAD needs multivariate and is very slow
    kw = {
        "window_size": 64, "stride": 64,
        "batch_size": 8, "threshold": 0.5,
        "min_spacing": None, "epochs": 500,
        "learning_rate": 0.001, "glad_iterations": 5,
        "eval_offset": 0.1, "axis": 0,
    }
    for thresh in [0.3, 0.5, 0.7]:
        kw_t = {**kw, "threshold": thresh}
        r = test_one("tGLAD", TGLADDetector, kw_t, multi)
        if r["ok"]:
            print(f"  multi th={thresh}: n={r['n']:>3d}, MAE={r['mae']:>6.1f}, CPs={r['cps'][:6]}, t={r['t']:.1f}s")
        else:
            print(f"  multi th={thresh}: ERR: {r['err'][:80]}, t={r['t']:.1f}s")
except ImportError as e:
    print(f"  SKIPPED: {e}")

# ======================================================================
#  DONE
# ======================================================================
print("\n" + "="*72)
print("  DONE")
print("="*72)
