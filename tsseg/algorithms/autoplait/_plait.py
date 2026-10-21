"""Python translation of the AutoPlait C code of Y. Matsubara (June 18, 2014).

The functions follow the C files of the original code: ``plait.c`` (regime
splitting, :class:`_Plait`), ``cps.c`` (cut point search,
:func:`_cut_point_search`), ``segbox.c`` (segment sets, :class:`_SegBox`),
``baum.c`` and ``forbackward.c`` (Baum-Welch, :func:`_e_step`,
:func:`_m_step`), ``kmeans.c`` (:func:`_kmeans`), ``viterbi.c``
(:func:`_viterbi_log`), ``hmmutils.c`` (:func:`_reset_hmm`, :func:`_log_pdf`)
and ``tool.c`` (z-normalisation, code lengths). The C code draws its random
numbers with ``rand()`` after ``srand(1)``; :func:`_rand` reproduces the
generator of the GNU C library, so that the translation draws the same numbers
in the same order and returns the segmentation of the C binary built on Linux.
The series is read as ``float`` (float32) values and normalised in float32, as
the C code does.

Departures from the C code, which change nothing where the C code works:

- A constant channel is set to zero after the z-normalisation, where the C
  code divides by a zero standard deviation and segments ``NaN`` values.
- The C code keeps a pool of ``ln(n) + 20`` segment sets and exits with
  ``too small maxc`` when it is exhausted, holds at most ``n / 10`` segments
  per set (more overflow the heap), and counts as several time points a line
  of its input file longer than 252 characters. None of these limits exists
  here; ``_Plait.exceeds_c_limits`` tells whether the first two were reached.
- ``BaumWelch`` always runs the expectation-maximisation: its ``k == 1`` test
  reads an uninitialised variable, which holds a stack address in the binary
  (checked with gdb), never 1.
- When the description length decreases up to ``MAXK`` states,
  ``_estimateHMM`` leaves the number of states uninitialised; the translation
  keeps ``MAXK`` (``_Plait.n_maxk_fallbacks`` counts these cases, which did not
  occur on the series compared with the binary).
- The cut point search keeps back-pointers instead of copying the list of cut
  points of each state at each time step; the paths are the same.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# hmm.h
ZERO = 0.000001
ONE = 0.999999
INF = 1000000000
MAXITER = 10
VARMAX = INF
VARMIN = ZERO
MAXVAL = 1.0
LOG_ZERO = math.log(ZERO)
# autoplait.h
MAXK = 16
MAXBAUMN = 3
NSAMPLE = 10
# tool.h: bits of a float
FB = 4 * 8
# plait.c
INFER_ITER_MIN = 3
INFER_ITER_MAX = 10
MINK = 1
LM = 0.1
REGIME_R = 0.03
SEGMENT_R = 0.03
# baum.c, hmmutils.c, kmeans.c, viterbi.c
DELTA = 1
VARINI = 10
KMAXITER = 100
VITHUGE = 100000000000.0
RAND_MAX = 2147483647


# ----------------------------------------------------------------------------
# Random numbers: rand() of the GNU C library (TYPE_3 additive generator)
# ----------------------------------------------------------------------------


def _srand(seed: int = 1) -> np.ndarray:
    """State of ``rand()`` after ``srand(seed)``: 31 words and two indices."""
    words = [0] * 31
    words[0] = seed
    word = seed
    for i in range(1, 31):
        hi, lo = divmod(word, 127773)
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        words[i] = word
    state = np.array(words + [3, 0], dtype=np.int64)
    for _ in range(310):
        _rand(state)
    return state


@njit(cache=True)
def _rand(state):
    """Next value of ``rand()``; ``state`` is updated in place."""
    front = state[31]
    rear = state[32]
    value = (state[front] + state[rear]) & 0xFFFFFFFF
    state[front] = value
    front += 1
    if front >= 31:
        front = 0
        rear += 1
    else:
        rear += 1
        if rear >= 31:
            rear = 0
    state[31] = front
    state[32] = rear
    return value >> 1


@njit(cache=True)
def _getrand(state):
    """``getrand()`` of ``tool.c``: ``rand() / RAND_MAX``."""
    return _rand(state) / RAND_MAX


@njit(cache=True)
def _skip_rand(state, count):
    for _ in range(count):
        _rand(state)


# ----------------------------------------------------------------------------
# HMM (hmmutils.c)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _reset_hmm(K, D, pi, A, mean, var, sum_w, M2, A_denom, state):
    """``ResetHMM``: random transitions, initial probabilities and means."""
    for i in range(K):
        total = 0.0
        for j in range(K):
            A[i, j] = _getrand(state)
            total += A[i, j]
        for j in range(K):
            A[i, j] /= total
    total = 0.0
    for i in range(K):
        pi[i] = _getrand(state)
        total += pi[i]
    for i in range(K):
        pi[i] /= total
    for j in range(K):
        for k in range(D):
            mean[j, k] = _getrand(state) * MAXVAL
            var[j, k] = VARINI
    for i in range(K):
        A_denom[i] = 0.0
        for k in range(D):
            sum_w[i, k] = 0.0
            M2[i, k] = 0.0


@njit(cache=True)
def _log_pdf(mean, var, s, obs, t, d):
    """``pdfL``: log-density of ``obs[t]`` in state ``s`` (diagonal Gaussian)."""
    p = 0.0
    for i in range(d):
        v = abs(var[s, i])
        diff = np.float64(obs[t, i]) - mean[s, i]
        g = math.exp(-diff * diff / (2 * v)) / math.sqrt(2 * math.pi * v)
        if g >= 1.0:
            g = ONE
        if g <= 0.0:
            g = ZERO
        p += math.log(ZERO + g)
    if p < LOG_ZERO:
        p = LOG_ZERO
    return p


class _HMM:
    """Parameters of an HMM with ``MAXK`` slots (``HMM`` of ``hmm.h``)."""

    __slots__ = (
        "k",
        "d",
        "n",
        "pi",
        "A",
        "mean",
        "var",
        "sum_w",
        "M2",
        "pi_denom",
        "A_denom",
    )

    def __init__(self, d):
        self.k = MAXK
        self.d = d
        self.n = 0
        self.pi = np.zeros(MAXK)
        self.A = np.zeros((MAXK, MAXK))
        self.mean = np.zeros((MAXK, d))
        self.var = np.zeros((MAXK, d))
        self.sum_w = np.zeros((MAXK, d))
        self.M2 = np.zeros((MAXK, d))
        self.pi_denom = np.zeros(1)
        self.A_denom = np.zeros(MAXK)

    def reset(self, k, state):
        """``ResetHMM(phmm, k, d)``."""
        self.n = 0
        self.k = k
        _reset_hmm(
            k,
            self.d,
            self.pi,
            self.A,
            self.mean,
            self.var,
            self.sum_w,
            self.M2,
            self.A_denom,
            state,
        )
        self.pi_denom[0] = 0.0

    def copy_from(self, other):
        """``SubstHMM(other, self)``."""
        self.k = other.k
        self.n = other.n
        for name in ("pi", "A", "mean", "var", "sum_w", "M2", "pi_denom", "A_denom"):
            np.copyto(getattr(self, name), getattr(other, name))

    def clone(self):
        new = _HMM(self.d)
        new.copy_from(self)
        return new


# ----------------------------------------------------------------------------
# Viterbi (viterbi.c)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _viterbi_log(obs, st, m, k, d, pi, A, mean, var):
    """``ViterbiL``: log-likelihood of the best state path of ``obs[st:st+m]``."""
    if m == 0:
        return 0.0
    piL = np.empty(k)
    AL = np.empty((k, k))
    for i in range(k):
        piL[i] = math.log(pi[i] + ZERO)
    for i in range(k):
        for j in range(k):
            AL[i, j] = math.log(A[i, j] + ZERO)
    prev = np.empty(k)
    cur = np.empty(k)
    for i in range(k):
        prev[i] = piL[i] + _log_pdf(mean, var, i, obs, st, d)
    for t in range(1, m):
        for j in range(k):
            maxval = -VITHUGE
            for i in range(k):
                val = prev[i] + AL[i, j]
                if val > maxval:
                    maxval = val
            cur[j] = maxval + _log_pdf(mean, var, j, obs, st + t, d)
        prev, cur = cur, prev
    lh = -VITHUGE
    found = False
    for i in range(k):
        if prev[i] > lh:
            lh = prev[i]
            found = True
    if not found:
        raise RuntimeError("AutoPlait: cannot compute viterbi path (LOG)")
    return lh


# ----------------------------------------------------------------------------
# Cut point search (cps.c)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _find_max(P, k):
    loc = -1
    best = -INF
    for i in range(k):
        if best < P[i]:
            best = P[i]
            loc = i
    return loc


@njit(cache=True)
def _cut_point_search(
    obs, st, length, d, k0, pi0, A0, mean0, var0, d0, k1, pi1, A1, mean1, var1, d1
):
    """``_search_aux``: split ``obs[st:st+length]`` between two regimes.

    Returns the likelihood term of ``CPSearch`` (``-lh / log 2``), the first
    index of each segment and the regime (0 or 1) of each segment. A
    back-pointer per time step and state replaces the path lists of the C code;
    the paths are the same.
    """
    if d0 <= 0 or d1 <= 0:
        raise RuntimeError("AutoPlait: delta == 0 (cpsearch)")
    logd0 = math.log(d0)
    logd1 = math.log(d1)
    log1d0 = math.log(1.0 - d0)
    log1d1 = math.log(1.0 - d1)
    lpi0 = np.empty(k0)
    lA0 = np.empty((k0, k0))
    lpi1 = np.empty(k1)
    lA1 = np.empty((k1, k1))
    for u in range(k0):
        lpi0[u] = math.log(pi0[u] + ZERO)
        for v in range(k0):
            lA0[u, v] = math.log(A0[u, v] + ZERO)
    for i in range(k1):
        lpi1[i] = math.log(pi1[i] + ZERO)
        for j in range(k1):
            lA1[i, j] = math.log(A1[i, j] + ZERO)
    # back-pointers: predecessor state, + MAXK if the regime switched
    back0 = np.zeros((length, k0), dtype=np.int8)
    back1 = np.zeros((length, k1), dtype=np.int8)
    Pv = np.empty(k0)
    Pu = np.empty(k0)
    Pj = np.empty(k1)
    Pi = np.empty(k1)
    b0 = np.empty(k0)
    b1 = np.empty(k1)

    t = st
    for v in range(k0):
        Pv[v] = logd1 + lpi0[v] + _log_pdf(mean0, var0, v, obs, t, d)
    for j in range(k1):
        Pj[j] = logd0 + lpi1[j] + _log_pdf(mean1, var1, j, obs, t, d)

    for tt in range(1, length):
        t = st + tt
        for u in range(k0):
            b0[u] = _log_pdf(mean0, var0, u, obs, t, d)
        for i in range(k1):
            b1[i] = _log_pdf(mean1, var1, i, obs, t, d)
        # Pu(t)
        maxj = _find_max(Pj, k1)
        for u in range(k0):
            maxPj = Pj[maxj] + logd1 + lpi0[u] + b0[u]
            maxPv = -INF
            maxv = 0
            for v in range(k0):
                val = log1d0 + Pv[v] + lA0[v, u] + b0[u]
                if val > maxPv:
                    maxPv = val
                    maxv = v
            if maxPj > maxPv:
                Pu[u] = maxPj
                back0[tt, u] = maxj + MAXK
            else:
                Pu[u] = maxPv
                back0[tt, u] = maxv
        # Pi(t)
        maxv = _find_max(Pv, k0)
        for i in range(k1):
            maxPv = Pv[maxv] + logd0 + lpi1[i] + b1[i]
            maxPj = -INF
            maxj2 = 0
            for j in range(k1):
                val = log1d1 + Pj[j] + lA1[j, i] + b1[i]
                if val > maxPj:
                    maxPj = val
                    maxj2 = j
            if maxPv > maxPj:
                Pi[i] = maxPv
                back1[tt, i] = maxv + MAXK
            else:
                Pi[i] = maxPj
                back1[tt, i] = maxj2
        Pu, Pv = Pv, Pu
        Pi, Pj = Pj, Pi

    maxv = _find_max(Pv, k0)
    maxj = _find_max(Pj, k1)
    if Pv[maxv] > Pj[maxj]:
        regime = 0
        state = maxv
        lh = Pv[maxv]
    else:
        regime = 1
        state = maxj
        lh = Pj[maxj]
    # walk the back-pointers: switch times, latest first
    switches = np.empty(length, dtype=np.int64)
    n_switches = 0
    for tt in range(length - 1, 0, -1):
        code = back0[tt, state] if regime == 0 else back1[tt, state]
        if code >= MAXK:
            switches[n_switches] = st + tt
            n_switches += 1
            regime = 1 - regime
            state = code - MAXK
        else:
            state = code
    starts = np.empty(n_switches + 1, dtype=np.int64)
    regimes = np.empty(n_switches + 1, dtype=np.int64)
    starts[0] = st
    regimes[0] = regime
    for i in range(n_switches):
        starts[i + 1] = switches[n_switches - 1 - i]
        regimes[i + 1] = 1 - regimes[i]
    return -lh / math.log(2.0), starts, regimes


# ----------------------------------------------------------------------------
# Baum-Welch (baum.c, forbackward.c)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _e_step(
    obs, seq_st, seq_m, offs, k, d, pi, A, mean, var, alpha, beta, gamma, B, xinorm
):
    """Forward, backward, gamma and the normaliser of xi for each sequence.

    Returns the sum of the forward log-likelihoods. ``xi`` itself is not
    stored: :func:`_m_step` recomputes each entry from ``alpha``, ``beta``,
    ``B`` and ``xinorm`` with the same operations as ``_computeXi``.
    """
    total = 0.0
    for r in range(seq_st.shape[0]):
        st = seq_st[r]
        m = seq_m[r]
        o = offs[r]
        for t in range(m):
            for i in range(k):
                B[o + t, i] = math.exp(_log_pdf(mean, var, i, obs, st + t, d))
        # Forward
        scale = np.zeros(m)
        for i in range(k):
            alpha[o, i] = pi[i] * B[o, i]
            scale[0] += alpha[o, i]
        for i in range(k):
            alpha[o, i] /= scale[0]
        for t in range(m - 1):
            scale[t + 1] = 0.0
            for j in range(k):
                acc = 0.0
                for i in range(k):
                    acc += alpha[o + t, i] * A[i, j]
                alpha[o + t + 1, j] = acc * B[o + t + 1, j]
                scale[t + 1] += alpha[o + t + 1, j]
            for j in range(k):
                alpha[o + t + 1, j] /= scale[t + 1]
        L = 0.0
        for t in range(m):
            L += math.log(scale[t])
        # Backward
        for i in range(k):
            beta[o + m - 1, i] = 1.0 / scale[m - 1]
        for t in range(m - 2, -1, -1):
            for i in range(k):
                acc = 0.0
                for j in range(k):
                    acc += A[i, j] * B[o + t + 1, j] * beta[o + t + 1, j]
                beta[o + t, i] = acc / scale[t]
        # gamma
        for t in range(m):
            acc = 0.0
            for j in range(k):
                gamma[o + t, j] = alpha[o + t, j] * beta[o + t, j]
                acc += gamma[o + t, j]
            for i in range(k):
                gamma[o + t, i] = gamma[o + t, i] / acc
        # normaliser of xi
        for t in range(m - 1):
            acc = 0.0
            for i in range(k):
                for j in range(k):
                    acc += (alpha[o + t, i] * beta[o + t + 1, j]) * (
                        A[i, j] * B[o + t + 1, j]
                    )
            xinorm[o + t] = acc
        total += L
    return total


@njit(cache=True)
def _m_step(
    obs,
    seq_st,
    seq_m,
    offs,
    k,
    d,
    pi,
    A,
    mean,
    var,
    sum_w,
    M2,
    pi_denom,
    A_denom,
    alpha,
    beta,
    gamma,
    B,
    xinorm,
    A_e,
):
    """``_computeParams``: incremental update of the HMM parameters.

    ``A_e`` is the transition matrix of the E-step that produced ``alpha``,
    ``beta``, ``gamma`` and ``B``.
    """
    n = seq_st.shape[0]
    # initial probability vector pi
    for i in range(k):
        pi[i] *= pi_denom[0]
    for i in range(k):
        for r in range(n):
            pi[i] += ZERO + gamma[offs[r], i]
    for i in range(k):
        if pi[i] < 0:
            pi[i] = 0.0
    denom = 0.0
    for i in range(k):
        denom += pi[i]
    pi_denom[0] = denom
    for i in range(k):
        pi[i] = pi[i] / denom
    # transition matrix A
    for i in range(k):
        for j in range(k):
            A[i, j] *= A_denom[i]
    for i in range(k):
        for j in range(k):
            for r in range(n):
                o = offs[r]
                for t in range(seq_m[r] - 1):
                    xi = (
                        (alpha[o + t, i] * beta[o + t + 1, j])
                        * (A_e[i, j] * B[o + t + 1, j])
                    ) / xinorm[o + t]
                    A[i, j] += ZERO + xi
    for i in range(k):
        for j in range(k):
            if A[i, j] < 0:
                A[i, j] = 0.0
    for i in range(k):
        A_denom[i] = 0.0
        for j in range(k):
            A_denom[i] += A[i, j]
    for i in range(k):
        for j in range(k):
            A[i, j] = A[i, j] / A_denom[i]
    # weighted incremental mean and variance
    for i in range(k):
        for c in range(d):
            if sum_w[i, c] == 0:
                mean[i, c] = 0.0
                M2[i, c] = 0.0
            for r in range(n):
                st = seq_st[r]
                o = offs[r]
                for t in range(seq_m[r]):
                    x = np.float64(obs[st + t, c])
                    w = gamma[o + t, i]
                    tmp = w + sum_w[i, c] + ZERO
                    delta = x - mean[i, c]
                    R = (delta * w) / tmp
                    mean[i, c] += R
                    M2[i, c] += sum_w[i, c] * delta * R
                    sum_w[i, c] = tmp
            var[i, c] = M2[i, c] / sum_w[i, c]
            if var[i, c] > VARMAX:
                var[i, c] = VARMAX
            if var[i, c] < VARMIN:
                var[i, c] = VARMIN


# ----------------------------------------------------------------------------
# k-means initialisation (kmeans.c)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _rand_int(m, state):
    r = int(_getrand(state) * m)
    if m == 0:
        return 0
    return r


@njit(cache=True)
def _calc_mean(obs, seq_st, seq_m, offs, idx, d, mean, row, label, state):
    for c in range(d):
        cnt = 0
        acc = 0.0
        for r in range(seq_st.shape[0]):
            for i in range(seq_m[r]):
                if label == -1 or idx[offs[r] + i] == label:
                    acc += np.float64(obs[seq_st[r] + i, c])
                    cnt += 1
        acc /= cnt + ZERO
        if cnt == 0:
            acc = _getrand(state)
        mean[row, c] = acc


@njit(cache=True)
def _calc_var(obs, seq_st, seq_m, offs, idx, d, var, row, label):
    for c in range(d):
        cnt = 0
        acc = 0.0
        acc_sqr = 0.0
        for r in range(seq_st.shape[0]):
            for i in range(seq_m[r]):
                if label == -1 or idx[offs[r] + i] == label:
                    x = obs[seq_st[r] + i, c]
                    cnt += 1
                    acc += np.float64(x)
                    # float product, as in C
                    acc_sqr += np.float64(x * x)
        mu = acc / (cnt + ZERO)
        v = ZERO + (acc_sqr - acc * mu) / (cnt - 1 + ZERO)
        if cnt <= 1 or v < VARMIN:
            v = VARMIN
        var[row, c] = v


@njit(cache=True)
def _kmeans(obs, seq_st, seq_m, offs, d, k, mean, var, state):
    """``Kmeans``: initial means and variances of the ``k`` states."""
    n = seq_st.shape[0]
    total_len = offs[n - 1] + seq_m[n - 1]
    idx = np.zeros(total_len, dtype=np.int64)
    if k == 1:
        _calc_mean(obs, seq_st, seq_m, offs, idx, d, mean, 0, -1, state)
        _calc_var(obs, seq_st, seq_m, offs, idx, d, var, 0, -1)
        return
    cnt = 0
    for r in range(n):
        for i in range(seq_m[r]):
            idx[offs[r] + i] = cnt % k
            cnt += 1
    prev = float(INF)
    for _ in range(KMAXITER):
        total = 0.0
        for r in range(n):
            for i in range(seq_m[r]):
                best = float(INF)
                for j in range(k):
                    dist = 0.0
                    for c in range(d):
                        diff = np.float64(obs[seq_st[r] + i, c]) - mean[j, c]
                        dist += diff * diff
                    if best > dist:
                        idx[offs[r] + i] = j
                        best = dist
                total += best
        # avoid single core
        for j in range(k):
            found = False
            for p in range(total_len):
                if idx[p] == j:
                    found = True
                    break
            if not found:
                r = _rand_int(n, state)
                mm = _rand_int(seq_m[r], state)
                idx[offs[r] + mm] = j
        for j in range(k):
            _calc_mean(obs, seq_st, seq_m, offs, idx, d, mean, j, j, state)
        if prev == total:
            break
        prev = total
    for j in range(k):
        _calc_mean(obs, seq_st, seq_m, offs, idx, d, mean, j, j, state)
        _calc_var(obs, seq_st, seq_m, offs, idx, d, var, j, j)


# ----------------------------------------------------------------------------
# Z-normalisation (tool.c)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _znorm(obs):
    """``ZnormSequence`` in float32 arithmetic, in place.

    A constant channel is set to zero (the C code divides by zero).
    """
    m, d = obs.shape
    for c in range(d):
        mean = np.float32(0.0)
        std = np.float32(0.0)
        for j in range(m):
            mean = np.float32(mean + obs[j, c])
        mean = np.float32(mean / np.float32(m))
        for j in range(m):
            diff = np.float32(obs[j, c] - mean)
            std = np.float32(np.float64(std) + math.pow(np.float64(diff), 2.0))
        std = np.float32(math.sqrt(np.float64(np.float32(std / np.float32(m)))))
        for j in range(m):
            if std == 0:
                obs[j, c] = np.float32(0.0)
            else:
                obs[j, c] = np.float32(
                    MAXVAL * np.float64(np.float32(obs[j, c] - mean)) / np.float64(std)
                )


# ----------------------------------------------------------------------------
# Code lengths (tool.c)
# ----------------------------------------------------------------------------


def _log_2(x):
    return math.log(float(x)) / math.log(2.0) if x > 0 else -math.inf


def _log_s(x):
    return 2.0 * _log_2(x) + 1.0


def _cost_hmm(k, d):
    return float(FB) * (k + k * k + 2 * k * d) + _log_s(k)


# ----------------------------------------------------------------------------
# Segment sets (segbox.c)
# ----------------------------------------------------------------------------


class _SegBox:
    """Set of segments of one regime and its HMM (``SegBox`` of ``autoplait.h``)."""

    __slots__ = ("st", "ln", "len", "costT", "costC", "delta", "label", "model")

    def __init__(self, d):
        self.st = []
        self.ln = []
        self.len = 0
        self.costT = float(INF)
        self.costC = float(INF)
        self.delta = 0.0
        self.label = ""
        self.model = _HMM(d)

    @property
    def n_segments(self):
        return len(self.st)

    def reset(self):
        """``ResetStEd``."""
        self.st = []
        self.ln = []
        self.len = 0
        self.costC = float(INF)
        self.costT = float(INF)

    def remove(self, i):
        """``RemoveStEd``."""
        self.len -= self.ln[i]
        del self.st[i]
        del self.ln[i]

    def _rm_overlap(self):
        curr = INF
        while curr > self.n_segments:
            curr = self.n_segments
            i = 0
            while i < self.n_segments - 1:
                st0 = self.st[i]
                ed0 = st0 + self.ln[i]
                st1 = self.st[i + 1]
                ed1 = st1 + self.ln[i + 1]
                ed = ed0 if ed0 > ed1 else ed1
                if ed0 + 1 >= st1:
                    self.remove(i + 1)
                    self.ln[i] = ed - st0
                i += 1

    def add(self, st, length):
        """``AddStEd``: insert a segment in start order, then merge overlaps."""
        if length <= 0:
            return
        if st < 0:
            st = 0
        loc = self.n_segments
        for i in range(self.n_segments):
            if self.st[i] > st:
                loc = i
                break
        self.st.insert(loc, st)
        self.ln.insert(loc, length)
        self._rm_overlap()
        self.len = sum(self.ln)

    def add_ex(self, st, length):
        """``AddStEd_ex``: append a segment, overlaps allowed."""
        self.len += length
        self.st.append(st)
        self.ln.append(length)

    def copy_to(self, other):
        """``CopyStEd``: segments, costs, delta and label (not the model)."""
        other.st = list(self.st)
        other.ln = list(self.ln)
        other.len = self.len
        other.costT = self.costT
        other.costC = self.costC
        other.delta = self.delta
        other.label = self.label

    def find_max(self):
        """``FindMaxStEd``: index of the first longest segment."""
        loc = -1
        best = -INF
        for i, length in enumerate(self.ln):
            if length > best:
                loc = i
                best = length
        return loc


# ----------------------------------------------------------------------------
# AutoPlait (plait.c)
# ----------------------------------------------------------------------------


class _Plait:
    """Workspace and main loop of AutoPlait (``PlaitWS`` and ``plait.c``).

    Parameters
    ----------
    X : np.ndarray
        Series of shape ``(n_timepoints, n_channels)``.

    Attributes
    ----------
    regimes : list of _SegBox
        The regimes found, in the order of the C code's output files.
    costT : float
        Total description length of the segmentation.
    trace : list of tuple
        ``(regimes, segments, cost)`` at each step of the main loop (the
        ``r|m|Cost`` lines printed by the C binary).
    exceeds_c_limits : bool
        Whether the C binary would have run out of segment sets (``too small
        maxc``) or overflowed a segment set on this series.
    n_maxk_fallbacks : int
        Number of model estimations whose description length decreased up to
        ``MAXK`` states (where the C code reads an uninitialised variable).
    """

    def __init__(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]
        m, d = X.shape
        if m < 2:
            raise ValueError("AutoPlait needs at least two time points.")
        self.m = m
        self.d = d
        self.lmax = m
        self.seedlen = int(math.ceil(self.lmax * LM))
        # Rows past the end of the series are zero, as in the C code, which
        # allocates 10**7 rows with calloc: the seeds of FixedSampling can
        # extend past the end of the series.
        obs = np.zeros((m + self.seedlen + 1, d), dtype=np.float32)
        obs[:m] = X
        _znorm(obs[:m])
        self.obs = obs
        self.state = _srand(1)
        # AllocPlait: InitHMM of the Baum-Welch copy, of the maxc segment sets
        # and of the sampling set draws K*K + K + K*d random numbers each.
        self.maxc = int(math.log(self.lmax) + 20)
        self.maxseg = int(self.lmax * 0.1)
        _skip_rand(self.state, (self.maxc + 2) * (MAXK * MAXK + MAXK + MAXK * d))
        self.in_use = 0
        self.exceeds_c_limits = False
        self.n_maxk_fallbacks = 0
        self.costT = float(INF)
        self.trace = []
        self.regimes = []

    # --- segment set pool ---------------------------------------------------

    def _get_s(self, parent, label):
        """``_getS``."""
        self.in_use += 1
        if self.in_use > self.maxc:
            self.exceeds_c_limits = True
        s = _SegBox(self.d)
        s.model.reset(MAXK, self.state)
        s.label = parent + label
        s.delta = 1.0 / float(self.lmax)
        return s

    def _release_s(self, s):
        self.in_use -= 1

    # --- model estimation ------------------------------------------------------

    def _baum_welch(self, hmm, seq_st, seq_m):
        """``BaumWelch`` with ``wantKMEANS``."""
        n = seq_st.shape[0]
        if n == 0:
            raise RuntimeError("AutoPlait: estimation error! n==0 (BaumWelch)")
        obs, state, d = self.obs, self.state, self.d
        offs = np.zeros(n, dtype=np.int64)
        offs[1:] = np.cumsum(seq_m)[:-1]
        if hmm.n == 0:
            hmm.reset(hmm.k, state)
            _kmeans(obs, seq_st, seq_m, offs, d, hmm.k, hmm.mean, hmm.var, state)
        prev_n = hmm.n
        hmm.n = n
        chmm = hmm.clone()
        k = hmm.k
        total_len = int(seq_m.sum())
        alpha = np.empty((total_len, k))
        beta = np.empty((total_len, k))
        gamma = np.empty((total_len, k))
        B = np.empty((total_len, k))
        xinorm = np.empty(total_len)

        def e_step():
            lsum = _e_step(
                obs,
                seq_st,
                seq_m,
                offs,
                k,
                d,
                hmm.pi,
                hmm.A,
                hmm.mean,
                hmm.var,
                alpha,
                beta,
                gamma,
                B,
                xinorm,
            )
            return lsum, hmm.A[:k, :k].copy()

        def m_step(A_e):
            _m_step(
                obs,
                seq_st,
                seq_m,
                offs,
                k,
                d,
                hmm.pi,
                hmm.A,
                hmm.mean,
                hmm.var,
                hmm.sum_w,
                hmm.M2,
                hmm.pi_denom,
                hmm.A_denom,
                alpha,
                beta,
                gamma,
                B,
                xinorm,
                A_e,
            )

        lsum, A_e = e_step()
        lpreb = lsum
        it = 0
        while True:
            hmm.copy_from(chmm)
            m_step(A_e)
            lsum, A_e = e_step()
            delta = lpreb - lsum
            lpreb = lsum
            it += 1
            if not (abs(delta) > DELTA and it < MAXITER):
                break
        m_step(A_e)
        if math.isnan(lsum):
            hmm.reset(hmm.k, state)
        hmm.n = hmm.n + prev_n
        return lsum

    def _estimate_hmm_k(self, s, k):
        """``_estimateHMM_k``."""
        k = min(max(k, MINK), MAXK)
        n = min(s.n_segments, MAXBAUMN)
        seq_st = np.empty(n, dtype=np.int64)
        seq_m = np.empty(n, dtype=np.int64)
        for i in range(n):
            # Split
            st = max(s.st[i], 0)
            length = s.ln[i]
            if self.m < st + length:
                length = self.m - st
            seq_st[i] = st
            seq_m[i] = length
        s.model.k = k
        s.model.n = 0
        s.model.reset(k, self.state)
        self._baum_welch(s.model, seq_st, seq_m)
        s.delta = float(s.n_segments) / float(s.len)

    def _estimate_hmm(self, s):
        """``_estimateHMM``: number of states by description length."""
        s.costT = float(INF)
        optk = MAXK
        for k in range(MINK, MAXK + 1):
            prev = s.costT
            self._estimate_hmm_k(s, k)
            self._compute_lh_mdl(s)
            if s.costT > prev:
                optk = k - 1
                break
        else:
            self.n_maxk_fallbacks += 1
        optk = min(max(optk, MINK), MAXK)
        self._estimate_hmm_k(s, optk)
        self._compute_lh_mdl(s)

    # --- costs -----------------------------------------------------------------

    def _viterbi(self, model, delta, st, length):
        """``_viterbi``: coding cost of ``obs[st:st+length]`` in one regime."""
        if st + length > self.obs.shape[0]:
            raise RuntimeError("AutoPlait: segment past the padded series.")
        lh = _viterbi_log(
            self.obs,
            st,
            length,
            model.k,
            self.d,
            model.pi,
            model.A,
            model.mean,
            model.var,
        )
        if delta <= 0 or delta >= 1:
            raise RuntimeError("AutoPlait: not appropriate delta (_computeLhMDL)")
        lh += math.log(delta)
        lh += (length - 1) * math.log(1.0 - delta)
        return -lh / math.log(2.0)

    def _compute_lh_mdl(self, s):
        """``_computeLhMDL``."""
        if s.n_segments == 0:
            s.costC = float(INF)
            s.costT = float(INF)
            return
        s.costC = 0.0
        for st, length in zip(s.st, s.ln):
            s.costC += self._viterbi(s.model, s.delta, st, length)
        s.costT = self._mdl(s)

    def _mdl(self, s):
        """``_MDL``: description length of a regime."""
        k = s.model.k
        cost_m = _cost_hmm(k, s.model.d)
        cost_len = 0.0
        for length in s.ln:
            cost_len += _log_2(length)
        cost_len += s.len * _log_2(k)
        return s.costC + cost_len + cost_m

    def _mdl_total(self, opt, cand):
        """``_MDLtotal``: description length of the whole segmentation."""
        r = len(opt) + len(cand)
        m = sum(s.n_segments for s in opt) + sum(s.n_segments for s in cand)
        cost = 0.0
        for s in opt:
            cost += s.costT
        cost_c = 0.0
        for s in cand:
            cost_c += s.costT
        cost += cost_c
        cost_t = cost + _log_s(r) + _log_s(m) + m * _log_2(r) + FB * r * r
        self.trace.append((r, m, cost_t))
        return cost_t

    # --- cut point search and pruning -------------------------------------------

    def _cps_search(self, Sx, s0, s1):
        """``CPSearch``: split the segments of ``Sx`` between ``s0`` and ``s1``."""
        s0.reset()
        s1.reset()
        m0, m1 = s0.model, s1.model
        lh = 0.0
        for st, length in zip(list(Sx.st), list(Sx.ln)):
            cost, starts, regimes = _cut_point_search(
                self.obs,
                st,
                length,
                self.d,
                m0.k,
                m0.pi,
                m0.A,
                m0.mean,
                m0.var,
                s0.delta,
                m1.k,
                m1.pi,
                m1.A,
                m1.mean,
                m1.var,
                s1.delta,
            )
            ends = list(starts[1:]) + [st + length]
            for a, b, regime in zip(starts, ends, regimes):
                (s0 if regime == 0 else s1).add(int(a), int(b - a))
            lh += cost
        if max(s0.n_segments, s1.n_segments) > self.maxseg:
            self.exceeds_c_limits = True
        return lh

    def _find_min_diff(self, s0, s1):
        """``_findMinDiff``: segment of ``s0`` that ``s1`` codes best."""
        best = float(INF)
        loc = -1
        for i, (st, length) in enumerate(zip(s0.st, s0.ln)):
            cost0 = self._viterbi(s0.model, s0.delta, st, length)
            cost1 = self._viterbi(s1.model, s1.delta, st, length)
            diff = cost1 - cost0
            if best > diff:
                loc = i
                best = diff
        return loc, best

    def _scan_min_diff(self, Sx, s0, s1):
        """``_scanMinDiff``."""
        loc0, _ = self._find_min_diff(s0, s1)
        loc1, _ = self._find_min_diff(s1, s0)
        if loc0 == -1 or loc1 == -1:
            return float(INF)
        tmp0 = self._get_s("", "")
        tmp1 = self._get_s("", "")
        tmp0.add(s0.st[loc0], s0.ln[loc0])
        tmp1.add(s1.st[loc1], s1.ln[loc1])
        self._estimate_hmm_k(tmp0, MINK)
        self._estimate_hmm_k(tmp1, MINK)
        cost = self._cps_search(Sx, tmp0, tmp1)
        self._release_s(tmp0)
        self._release_s(tmp1)
        return cost

    def _remove_noise_aux(self, Sx, s0, s1, per):
        """``_removeNoise_aux``: move the segments that the other regime codes
        better, by less than ``per`` of the total cost."""
        if per == 0:
            return
        mprev = INF
        th = self.costT * per
        while mprev > s0.n_segments + s1.n_segments:
            mprev = s0.n_segments + s1.n_segments
            loc0, diff0 = self._find_min_diff(s0, s1)
            loc1, diff1 = self._find_min_diff(s1, s0)
            if diff0 < diff1:
                best, which = diff0, 0
            else:
                best, which = diff1, 1
            if best < th:
                if which == 0:
                    s1.add(s0.st[loc0], s0.ln[loc0])
                    s0.remove(loc0)
                else:
                    s0.add(s1.st[loc1], s1.ln[loc1])
                    s1.remove(loc1)

    def _remove_noise(self, Sx, s0, s1):
        """``_removeNoise``: prune small segments at the best threshold."""
        if s0.n_segments <= 1 and s1.n_segments <= 1:
            return
        per = SEGMENT_R
        self._remove_noise_aux(Sx, s0, s1, per)
        cost = self._scan_min_diff(Sx, s0, s1)
        opt0 = self._get_s("", "")
        opt1 = self._get_s("", "")
        s0.copy_to(opt0)
        s1.copy_to(opt1)
        prev = float(INF)
        while per <= SEGMENT_R * 10:
            if cost >= INF:
                break
            per *= 2
            self._remove_noise_aux(Sx, s0, s1, per)
            if s0.n_segments <= 1 or s1.n_segments <= 1:
                break
            cost = self._scan_min_diff(Sx, s0, s1)
            if prev > cost:
                s0.copy_to(opt0)
                s1.copy_to(opt1)
            else:
                break
            prev = cost
        opt0.copy_to(s0)
        opt1.copy_to(s1)
        self._release_s(opt0)
        self._release_s(opt1)

    def _cps(self, Sx, s0, s1, remove_noise):
        """``_cps``."""
        self._cps_search(Sx, s0, s1)
        if remove_noise:
            self._remove_noise(Sx, s0, s1)
        self._compute_lh_mdl(s0)
        self._compute_lh_mdl(s1)

    # --- regime split ------------------------------------------------------------

    def _uniform_set(self, Sx, length, trial):
        """``UniformSet``: candidate seed positions."""
        # C integer division, truncated towards zero
        slide = int((Sx.len - length) / trial)
        U = []
        for st, seg_len in zip(Sx.st, Sx.ln):
            if len(U) >= trial:
                return U
            ed = st + seg_len
            for j in range(trial):
                nxt = st + j * slide
                if nxt + length > ed:
                    U.append(max(ed - length, 0))
                    break
                U.append(nxt)
        return U

    def _fixed_sampling(self, Sx, s0, s1, length):
        """``FixedSampling``."""
        s0.reset()
        s1.reset()
        loc = 0 % Sx.n_segments
        s0.add(Sx.st[loc], length)
        loc = 1 % Sx.n_segments
        s1.add(Sx.st[loc] + int(Sx.ln[loc] / 2), length)

    def _find_centroid(self, Sx, s0, s1, nsamples, seedlen):
        """``_findCentroid``: best pair of seeds among uniform samples."""
        cost_min = float(INF)
        best = None
        U = self._uniform_set(Sx, seedlen, nsamples)
        for i1 in range(len(U)):
            for i2 in range(i1 + 1, len(U)):
                # UniformSampling
                s0.reset()
                s1.reset()
                st0 = U[i1 % len(U)]
                st1 = U[i2 % len(U)]
                if abs(st0 - st1) < seedlen:
                    continue
                s0.add(st0, seedlen)
                s1.add(st1, seedlen)
                if s0.n_segments == 0 or s1.n_segments == 0:
                    continue
                current = (s0.st[0], s0.ln[0], s1.st[0], s1.ln[0])
                self._estimate_hmm_k(s0, MINK)
                self._estimate_hmm_k(s1, MINK)
                self._cps(Sx, s0, s1, True)
                if s0.n_segments == 0 or s1.n_segments == 0:
                    continue
                if cost_min > s0.costT + s1.costT:
                    cost_min = s0.costT + s1.costT
                    best = current
        if cost_min == INF:
            self._fixed_sampling(Sx, s0, s1, seedlen)
            return float(INF)
        s0.reset()
        s1.reset()
        s0.add(best[0], best[1])
        s1.add(best[2], best[3])
        return cost_min

    def _select_largest(self, s):
        """``_selectLargest``: keep the longest segment only."""
        i = s.find_max()
        st, length = s.st[i], s.ln[i]
        s.reset()
        s.add(st, length)

    def _regime_est_aux(self, Sx, s0, s1):
        """``_regimeEst_aux``: alternate model estimation and cut point search."""
        opt0 = self._get_s("", "")
        opt1 = self._get_s("", "")
        for i in range(INFER_ITER_MAX):
            self._select_largest(s0)
            self._select_largest(s1)
            self._estimate_hmm(s0)
            self._estimate_hmm(s1)
            self._cps(Sx, s0, s1, True)
            if s0.n_segments == 0 or s1.n_segments == 0:
                break
            diff = (opt0.costT + opt1.costT) - (s0.costT + s1.costT)
            if diff > 0:
                s0.copy_to(opt0)
                s1.copy_to(opt1)
            elif i >= INFER_ITER_MIN:
                break
        opt0.copy_to(s0)
        opt1.copy_to(s1)
        self._release_s(opt0)
        self._release_s(opt1)

    def _regime_split(self, Sx, s0, s1):
        """``_regimeSplit``: try to split regime ``Sx`` into ``s0`` and ``s1``."""
        self._find_centroid(Sx, s0, s1, NSAMPLE, self.seedlen)
        self._regime_est_aux(Sx, s0, s1)
        if s0.n_segments == 0 or s1.n_segments == 0:
            return
        self._estimate_hmm(s0)
        self._estimate_hmm(s1)

    def run(self):
        """``_plait``: split regimes while the description length decreases."""
        candidates = []
        optimal = []
        Sx = self._get_s("", "")
        Sx.costT = float(INF)
        Sx.add(0, self.m)
        self._estimate_hmm(Sx)
        candidates.append(Sx)
        while True:
            self.costT = self._mdl_total(optimal, candidates)
            if not candidates:
                break
            Sx = candidates.pop()
            s0 = self._get_s(Sx.label, "0")
            s1 = self._get_s(Sx.label, "1")
            self._regime_split(Sx, s0, s1)
            cost_s01 = s0.costT + s1.costT
            if cost_s01 + Sx.costT * REGIME_R < Sx.costT:
                candidates.append(s0)
                candidates.append(s1)
                self._release_s(Sx)
            else:
                optimal.append(Sx)
                self._release_s(s0)
                self._release_s(s1)
        self.regimes = optimal
        return self


def autoplait(X):
    """Run AutoPlait on a series.

    Parameters
    ----------
    X : array-like
        Series of shape ``(n_timepoints,)`` or ``(n_timepoints, n_channels)``.

    Returns
    -------
    _Plait
        The fitted workspace: ``regimes`` holds the segments of each regime.
    """
    return _Plait(X).run()
