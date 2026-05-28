#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Reconstruction Method, benchmark variant CReMB with dcGM topology.

Model
-----
This module implements the benchmark obtained by combining

1. a density-corrected Gravity Model (dcGM) for the binary layer, and
2. the CReMB weighted dressing for positive dyadic weights.

For every dyad ``(i, j)`` the binary link probability is

    p_ij(z) = (z s_i^{out} s_j^{in}) / (1 + z s_i^{out} s_j^{in}),

in directed mode, and

    p_ij(z) = (z s_i s_j) / (1 + z s_i s_j),

in undirected mode. The scalar ``z >= 0`` is calibrated by matching the
expected number of links.

Conditionally on ``a_ij = 1``, CReMB assigns an exponential weight with mean

    E[w_ij | a_ij = 1] = (s_i^{out} s_j^{in}) / (W p_ij),

where ``W = sum_i s_i^{out} = sum_j s_j^{in}`` in directed mode, and
``W = sum_i s_i`` in undirected mode. Therefore the unconditional dyadic mean is

    E[w_ij] = (s_i^{out} s_j^{in}) / W,

or ``s_i s_j / W`` in undirected mode.

Interface design
----------------
The public API mirrors the current ``wmsm.py`` helper functions so that the
same notebook structure can be reused with minimal changes. In particular,
``delta`` is accepted as the keyword name of the binary scale parameter for
compatibility, although the parameter is denoted by ``z`` in the present file.

Implementation notes
--------------------
- The binary calibration is solved in log-space, ``theta = log z``, by damped
  Newton iterations with an Armijo backtracking line search.
- Low-level numerical kernels are optionally accelerated with Numba. When
  Numba is unavailable, the module falls back to pure-Python/NumPy paths.
- Zero strengths imply zero binary probabilities for every incident dyad at any
  finite ``z``.
- The weighted unconditional expectation is independent of ``z``. This is a
  structural feature of CReMB rather than a numerical approximation.
- When diagonal dyads are excluded, the unconditional expected strengths are no
  longer guaranteed to coincide exactly with the supplied marginals. This is
  the same off-diagonal caveat encountered by any gravity expectation built by
  masking self-dyads after construction.

Primary references
------------------
Parisi, Squartini, Garlaschelli, "A faster horse on a safer trail: generalized
inference for the efficient reconstruction of weighted networks", arXiv:1811.09829.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Optional accelerator (Numba)
# -----------------------------------------------------------------------------
try:  # pragma: no cover
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*_args, **_kwargs):  # type: ignore
        """Return an identity decorator when Numba is unavailable."""
        def _decorator(func):
            return func
        return _decorator


Array = np.ndarray


# -----------------------------------------------------------------------------
# Low-level numerics
# -----------------------------------------------------------------------------
@njit(cache=True)
def _expit_scalar(t: float) -> float:
    """
    Logistic sigmoid ``sigma(t) = 1 / (1 + exp(-t))`` evaluated without overflow.
    """
    if t >= 0.0:
        e = np.exp(-t)
        return 1.0 / (1.0 + e)
    e = np.exp(t)
    return e / (1.0 + e)


@njit(cache=True)
def _log_strengths(s: Array) -> Array:
    """
    Map nonnegative strengths to logarithms, with ``log(0) = -inf``.
    """
    n = s.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        si = s[i]
        out[i] = np.log(si) if si > 0.0 else -np.inf
    return out


@njit(cache=True)
def _Lhat_and_dL_directed(theta: float, log_sout: Array, log_sin: Array, include_diagonal: bool) -> Tuple[float, float]:
    """
    Compute ``L_hat(theta)`` and ``dL_hat / dtheta`` in directed mode.
    """
    n = log_sout.size
    L_hat = 0.0
    dL = 0.0
    for i in range(n):
        for j in range(n):
            if (not include_diagonal) and (i == j):
                continue
            t = theta + log_sout[i] + log_sin[j]
            p = _expit_scalar(t)
            L_hat += p
            dL += p * (1.0 - p)
    return L_hat, dL


@njit(cache=True)
def _Lhat_and_dL_undirected(theta: float, log_s: Array, include_diagonal: bool) -> Tuple[float, float]:
    """
    Compute ``L_hat(theta)`` and ``dL_hat / dtheta`` in undirected mode.
    """
    n = log_s.size
    L_hat = 0.0
    dL = 0.0
    for i in range(n):
        jmax = i + 1 if include_diagonal else i
        for j in range(jmax):
            t = theta + log_s[i] + log_s[j]
            p = _expit_scalar(t)
            L_hat += p
            dL += p * (1.0 - p)
    return L_hat, dL


@njit(cache=True)
def _probabilities_directed(log_sout: Array, log_sin: Array, theta: float, include_diagonal: bool) -> Array:
    """
    Construct the directed CReMB binary-probability matrix.
    """
    n = log_sout.size
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if (not include_diagonal) and (i == j):
                continue
            t = theta + log_sout[i] + log_sin[j]
            P[i, j] = _expit_scalar(t)
    return P


@njit(cache=True)
def _probabilities_undirected(log_s: Array, theta: float, include_diagonal: bool) -> Array:
    """
    Construct the symmetric CReMB binary-probability matrix.
    """
    n = log_s.size
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        jmax = i + 1 if include_diagonal else i
        for j in range(jmax):
            t = theta + log_s[i] + log_s[j]
            p = _expit_scalar(t)
            P[i, j] = p
            P[j, i] = p
    return P


@njit(cache=True)
def _degrees_directed(log_sout: Array, log_sin: Array, theta: float, include_diagonal: bool) -> Tuple[Array, Array, float]:
    """
    Compute expected out-degrees, in-degrees, and expected link count in directed mode.
    """
    n = log_sout.size
    kout = np.zeros(n, dtype=np.float64)
    kin = np.zeros(n, dtype=np.float64)
    L_hat = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if (not include_diagonal) and (i == j):
                continue
            t = theta + log_sout[i] + log_sin[j]
            p = _expit_scalar(t)
            row_sum += p
            kin[j] += p
            L_hat += p
        kout[i] = row_sum
    return kout, kin, L_hat


@njit(cache=True)
def _degrees_undirected(log_s: Array, theta: float, include_diagonal: bool) -> Tuple[Array, Array, float]:
    """
    Compute expected degrees and expected link count in undirected mode.
    """
    n = log_s.size
    k = np.zeros(n, dtype=np.float64)
    L_hat = 0.0
    for i in range(n):
        jmax = i + 1 if include_diagonal else i
        for j in range(jmax):
            t = theta + log_s[i] + log_s[j]
            p = _expit_scalar(t)
            if i == j:
                k[i] += p
            else:
                k[i] += p
                k[j] += p
            L_hat += p
    return k, k.copy(), L_hat


@njit(cache=True)
def _expected_degrees_firm_streaming_directed(
    sout: Array,
    sin: Array,
    theta: float,
    chunk_size: int,
    include_diagonal: bool,
) -> Tuple[Array, Array, float]:
    """
    Streaming computation of expected degrees in directed mode.
    """
    n = sout.shape[0]
    log_sout = _log_strengths(sout)
    log_sin = _log_strengths(sin)
    kout = np.zeros(n, dtype=np.float64)
    kin = np.zeros(n, dtype=np.float64)
    L_hat = 0.0

    for j0 in range(0, n, chunk_size):
        j1 = j0 + chunk_size
        if j1 > n:
            j1 = n

        for i in range(n):
            row_acc = 0.0
            for j in range(j0, j1):
                if (not include_diagonal) and (i == j):
                    continue
                t = theta + log_sout[i] + log_sin[j]
                p = _expit_scalar(t)
                row_acc += p
                kin[j] += p
                L_hat += p
            kout[i] += row_acc

    return kout, kin, L_hat


@njit(cache=True)
def _expected_degrees_firm_streaming_undirected(
    s: Array,
    theta: float,
    chunk_size: int,
    include_diagonal: bool,
) -> Tuple[Array, Array, float]:
    """
    Streaming computation of expected degrees in undirected mode.
    """
    n = s.shape[0]
    log_s = _log_strengths(s)
    k = np.zeros(n, dtype=np.float64)
    L_hat = 0.0

    for i0 in range(0, n, chunk_size):
        i1 = i0 + chunk_size
        if i1 > n:
            i1 = n

        for i in range(i0, i1):
            jmax = i + 1 if include_diagonal else i
            for j in range(jmax):
                t = theta + log_s[i] + log_s[j]
                p = _expit_scalar(t)
                if i == j:
                    k[i] += p
                else:
                    k[i] += p
                    k[j] += p
                L_hat += p

    return k, k.copy(), L_hat


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def _validate_vector(x: Array, *, name: str) -> Array:
    """
    Validate and coerce a one-dimensional nonnegative float array.
    """
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return arr



def _validate_undirected_strengths(
    sout: Array,
    sin: Optional[Array],
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> Array:
    """
    Return the unique strength vector in undirected mode.
    """
    s = _validate_vector(sout, name="sout")
    if sin is None:
        return s
    t = _validate_vector(sin, name="sin")
    if s.shape != t.shape:
        raise ValueError("sout and sin must have the same length.")
    if not np.allclose(s, t, atol=atol, rtol=rtol):
        raise ValueError(
            "Undirected mode requires a single strength sequence. "
            "The provided out- and in-strength vectors are not equal."
        )
    return s



def _effective_L_max_directed(sout: Array, sin: Array, include_diagonal: bool) -> float:
    """
    Maximum achievable expected link count at finite ``z`` in directed mode.
    """
    active_out = sout > 0.0
    active_in = sin > 0.0
    if include_diagonal:
        return float(np.sum(active_out) * np.sum(active_in))
    count = 0.0
    n = sout.size
    for i in range(n):
        if not active_out[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if active_in[j]:
                count += 1.0
    return float(count)



def _effective_L_max_undirected(s: Array, include_diagonal: bool) -> float:
    """
    Maximum achievable expected link count at finite ``z`` in undirected mode.
    """
    m = int(np.sum(s > 0.0))
    if include_diagonal:
        return float(m * (m + 1) // 2)
    return float(m * (m - 1) // 2)



def _outer_weight_expectation_directed(sout: Array, sin: Array, include_diagonal: bool) -> Array:
    """
    Return the unconditional expected-weight matrix in directed mode.
    """
    total_weight = float(np.sum(sout))
    if total_weight <= 0.0:
        raise ValueError("The total weight implied by sout must be positive.")
    if not np.isclose(total_weight, float(np.sum(sin)), atol=1e-10, rtol=1e-10):
        raise ValueError("Directed CReMB requires sum(sout) == sum(sin).")
    Wexp = np.outer(sout, sin) / total_weight
    if not include_diagonal:
        np.fill_diagonal(Wexp, 0.0)
    return Wexp



def _outer_weight_expectation_undirected(s: Array, include_diagonal: bool) -> Array:
    """
    Return the unconditional expected-weight matrix in undirected mode.
    """
    total_weight = float(np.sum(s))
    if total_weight <= 0.0:
        raise ValueError("The total weight implied by the strength sequence must be positive.")
    Wexp = np.outer(s, s) / total_weight
    if not include_diagonal:
        np.fill_diagonal(Wexp, 0.0)
    return Wexp


# -----------------------------------------------------------------------------
# Public calibration routines
# -----------------------------------------------------------------------------
def fit_z_from_strengths(
    sout: Array,
    sin: Optional[Array],
    L_obs: float,
    *,
    include_diagonal: bool = True,
    directed: bool = True,
    tol_L: float = 1e-10,
    max_iter: int = 200,
    line_search: bool = True,
    theta0: Optional[float] = None,
) -> float:
    """
    Calibrate the binary CReMB scale ``z`` by matching the expected number of links.

    Parameters
    ----------
    sout, sin:
        Strength vectors. In undirected mode, ``sin`` may be omitted or must
        coincide with ``sout``.
    L_obs:
        Observed number of binary links at the chosen scale.
    include_diagonal:
        Whether diagonal dyads are included in the support.
    directed:
        Whether the network is directed.
    tol_L:
        Absolute tolerance on ``|L_hat - L_obs|``.
    max_iter:
        Maximum number of Newton iterations.
    line_search:
        If ``True``, use Armijo backtracking on the quadratic merit function.
    theta0:
        Optional initial guess for ``theta = log z``.
    """
    if not np.isfinite(L_obs) or L_obs < 0.0:
        raise ValueError("L_obs must be a finite non-negative scalar.")

    if directed:
        s_out = _validate_vector(sout, name="sout")
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        s_in = _validate_vector(sin, name="sin")
        if s_out.size != s_in.size:
            raise ValueError("sout and sin must have the same length.")
        n = int(s_out.size)
        if n < 1:
            raise ValueError("At least one node is required.")
        if L_obs <= tol_L:
            return 0.0
        L_max_eff = _effective_L_max_directed(s_out, s_in, include_diagonal=include_diagonal)
        if L_obs > L_max_eff + tol_L:
            raise ValueError(
                f"Infeasible L_obs={L_obs} given strengths: maximum achievable is {L_max_eff}."
            )
        if L_obs >= L_max_eff - tol_L:
            return 1e300
        log_sout = _log_strengths(s_out)
        log_sin = _log_strengths(s_in)
        expected_fn = lambda th: _Lhat_and_dL_directed(th, log_sout, log_sin, bool(include_diagonal))
        if theta0 is None:
            sum_products = float(np.sum(np.outer(s_out, s_in)))
            if not include_diagonal:
                sum_products -= float(np.dot(s_out, s_in))
            if sum_products <= 0.0 or not np.isfinite(sum_products):
                raise RuntimeError("Sum of dyadic strength products is non-positive.")
            z0 = max(1e-300, float(L_obs) / sum_products)
            theta = float(np.log(z0))
        else:
            if not np.isfinite(theta0):
                raise ValueError("theta0 must be finite if provided.")
            theta = float(theta0)
    else:
        s = _validate_undirected_strengths(sout, sin)
        n = int(s.size)
        if n < 1:
            raise ValueError("At least one node is required.")
        if L_obs <= tol_L:
            return 0.0
        L_max_eff = _effective_L_max_undirected(s, include_diagonal=include_diagonal)
        if L_obs > L_max_eff + tol_L:
            raise ValueError(
                f"Infeasible L_obs={L_obs} given strengths: maximum achievable is {L_max_eff}."
            )
        if L_obs >= L_max_eff - tol_L:
            return 1e300
        log_s = _log_strengths(s)
        expected_fn = lambda th: _Lhat_and_dL_undirected(th, log_s, bool(include_diagonal))
        if theta0 is None:
            sum_products = float(np.sum(np.outer(s, s)))
            if not include_diagonal:
                sum_products -= float(np.dot(s, s))
            else:
                sum_products = 0.5 * (sum_products + float(np.dot(s, s)))
            if sum_products <= 0.0 or not np.isfinite(sum_products):
                raise RuntimeError("Sum of dyadic strength products is non-positive.")
            z0 = max(1e-300, float(L_obs) / sum_products)
            theta = float(np.log(z0))
        else:
            if not np.isfinite(theta0):
                raise ValueError("theta0 must be finite if provided.")
            theta = float(theta0)

    for _ in range(int(max_iter)):
        L_hat, dL = expected_fn(theta)
        diff = float(L_hat - L_obs)

        if abs(diff) <= tol_L:
            return float(np.exp(theta))

        if (not np.isfinite(dL)) or dL <= 0.0:
            raise RuntimeError("Non-positive derivative encountered during CReMB calibration.")

        step = -diff / float(dL)

        if not line_search:
            theta = float(theta + step)
            continue

        phi_old = 0.5 * diff * diff
        alpha = 1.0
        c1 = 1e-4
        accepted = False

        for _ls in range(40):
            theta_trial = float(theta + alpha * step)
            L_trial, _ = expected_fn(theta_trial)
            diff_trial = float(L_trial - L_obs)
            phi_new = 0.5 * diff_trial * diff_trial
            if phi_new <= phi_old * (1.0 - c1 * alpha):
                theta = theta_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            theta = float(theta + 1e-3 * step)

    return float(np.exp(theta))



def fit_delta_from_strengths(
    sout: Array,
    sin: Optional[Array],
    L_obs: float,
    *,
    include_diagonal: bool = True,
    directed: bool = True,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Compatibility wrapper matching the current ``wmsm.py`` signature.

    The returned value is the CReMB binary scale ``z``. The keyword name
    ``delta`` is kept only to simplify notebook reuse.
    """
    return fit_z_from_strengths(
        sout,
        sin,
        L_obs,
        include_diagonal=include_diagonal,
        directed=directed,
        tol_L=float(tol) * max(float(L_obs), 1.0),
        max_iter=max_iter,
        line_search=True,
    )


# -----------------------------------------------------------------------------
# Public matrix/vector utilities
# -----------------------------------------------------------------------------
def probability_matrix_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Array:
    """
    Return the CReMB binary-probability matrix.

    The keyword ``delta`` is interpreted as the binary scale ``z`` for API
    compatibility with the existing wMSM notebook.
    """
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be a finite non-negative scalar.")

    if directed:
        s_out = _validate_vector(sout, name="sout")
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        s_in = _validate_vector(sin, name="sin")
        if s_out.size != s_in.size:
            raise ValueError("sout and sin must have the same length.")
        theta = float(np.log(delta)) if delta > 0.0 else -np.inf
        P = _probabilities_directed(_log_strengths(s_out), _log_strengths(s_in), theta, bool(include_diagonal))
        return P

    s = _validate_undirected_strengths(sout, sin)
    theta = float(np.log(delta)) if delta > 0.0 else -np.inf
    P = _probabilities_undirected(_log_strengths(s), theta, bool(include_diagonal))
    return P



def expected_degrees_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Tuple[Array, Array, float]:
    """
    Compute expected degrees and expected link count.
    """
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be a finite non-negative scalar.")

    theta = float(np.log(delta)) if delta > 0.0 else -np.inf

    if directed:
        s_out = _validate_vector(sout, name="sout")
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        s_in = _validate_vector(sin, name="sin")
        if s_out.size != s_in.size:
            raise ValueError("sout and sin must have the same length.")
        return _degrees_directed(_log_strengths(s_out), _log_strengths(s_in), theta, bool(include_diagonal))

    s = _validate_undirected_strengths(sout, sin)
    return _degrees_undirected(_log_strengths(s), theta, bool(include_diagonal))



def expected_weights_matrix_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: Optional[float] = None,
    rho: Optional[float] = None,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Array:
    """
    Return the unconditional expected-weight matrix.

    Parameters ``delta`` and ``rho`` are accepted for compatibility with the
    current wMSM notebook, but they are not used because the unconditional
    CReMB expectation depends only on the strength marginals.
    """
    if directed:
        s_out = _validate_vector(sout, name="sout")
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        s_in = _validate_vector(sin, name="sin")
        if s_out.size != s_in.size:
            raise ValueError("sout and sin must have the same length.")
        return _outer_weight_expectation_directed(s_out, s_in, include_diagonal=bool(include_diagonal))

    s = _validate_undirected_strengths(sout, sin)
    return _outer_weight_expectation_undirected(s, include_diagonal=bool(include_diagonal))



def conditional_expected_weights_matrix_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Array:
    """
    Return the conditional mean matrix ``E[w_ij | a_ij = 1]``.

    Dyads with zero binary probability are assigned value zero.
    """
    P = probability_matrix_from_strengths(
        sout,
        sin,
        delta=delta,
        include_diagonal=include_diagonal,
        directed=directed,
    )
    Wexp = expected_weights_matrix_from_strengths(
        sout,
        sin,
        delta=delta,
        include_diagonal=include_diagonal,
        directed=directed,
    )
    out = np.zeros_like(Wexp)
    positive = P > 0.0
    out[positive] = Wexp[positive] / P[positive]
    return out



def expected_degrees_firm(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    chunk_size: int = 1024,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Tuple[Array, Array, float]:
    """
    Memory-stable expected-degree computation for large arrays.

    This array-based function mirrors the firm-level streaming utility used in
    the wMSM workflow, but it does not require pandas inputs.
    """
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be a finite non-negative scalar.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    theta = float(np.log(delta)) if delta > 0.0 else -np.inf

    if directed:
        s_out = _validate_vector(sout, name="sout")
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        s_in = _validate_vector(sin, name="sin")
        if s_out.size != s_in.size:
            raise ValueError("sout and sin must have the same length.")
        return _expected_degrees_firm_streaming_directed(
            s_out,
            s_in,
            theta,
            int(chunk_size),
            bool(include_diagonal),
        )

    s = _validate_undirected_strengths(sout, sin)
    return _expected_degrees_firm_streaming_undirected(
        s,
        theta,
        int(chunk_size),
        bool(include_diagonal),
    )



def rho_from_total_weight(delta: float, total_weight: float) -> float:
    """
    Compatibility placeholder for the current wMSM notebook.

    CReMB does not require a geometric mark parameter. The function returns
    ``nan`` to make the quantity explicitly non-applicable while still allowing
    legacy notebook code to run unchanged if ``rho`` is only threaded through to
    ``expected_weights_matrix_from_strengths``.
    """
    _ = (delta, total_weight)
    return float("nan")


# -----------------------------------------------------------------------------
# Convenience fit wrapper and model container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CReMBResult:
    """
    Container for calibrated CReMB outputs.
    """
    z: float
    P: Array
    expected_out_degrees: Array
    expected_in_degrees: Array
    expected_links: float
    expected_weights: Array
    expected_out_strengths: Array
    expected_in_strengths: Array
    total_weight: float


class CReMBModel:
    """
    Small cache-oriented container for dcGM + CReMB calibration outputs.
    """

    def __init__(
        self,
        sout: Array,
        sin: Optional[Array] = None,
        *,
        L_obs: float,
        include_diagonal: bool = True,
        directed: bool = True,
    ) -> None:
        self.directed = bool(directed)
        self.include_diagonal = bool(include_diagonal)
        self.L_obs = float(L_obs)
        if not np.isfinite(self.L_obs) or self.L_obs < 0.0:
            raise ValueError("L_obs must be a finite non-negative scalar.")

        if self.directed:
            self.sout = _validate_vector(sout, name="sout")
            if sin is None:
                raise ValueError("sin must be provided in directed mode.")
            self.sin = _validate_vector(sin, name="sin")
            if self.sout.size != self.sin.size:
                raise ValueError("sout and sin must have the same length.")
        else:
            s = _validate_undirected_strengths(sout, sin)
            self.sout = s
            self.sin = s.copy()

        self._result: Optional[CReMBResult] = None

    def fit(
        self,
        *,
        tol_L: float = 1e-10,
        max_iter: int = 200,
        line_search: bool = True,
        theta0: Optional[float] = None,
    ) -> CReMBResult:
        """
        Calibrate ``z`` and cache all expected quantities.
        """
        z = fit_z_from_strengths(
            self.sout,
            self.sin,
            self.L_obs,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
            tol_L=tol_L,
            max_iter=max_iter,
            line_search=line_search,
            theta0=theta0,
        )
        P = probability_matrix_from_strengths(
            self.sout,
            self.sin,
            delta=z,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
        )
        kout, kin, EL = expected_degrees_from_strengths(
            self.sout,
            self.sin,
            delta=z,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
        )
        Wexp = expected_weights_matrix_from_strengths(
            self.sout,
            self.sin,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
        )
        sout_exp = Wexp.sum(axis=1)
        sin_exp = Wexp.sum(axis=0)
        total_weight = float(np.sum(self.sout))

        res = CReMBResult(
            z=float(z),
            P=P,
            expected_out_degrees=kout,
            expected_in_degrees=kin,
            expected_links=float(EL),
            expected_weights=Wexp,
            expected_out_strengths=sout_exp,
            expected_in_strengths=sin_exp,
            total_weight=total_weight,
        )
        self._result = res
        return res

    def result(self) -> CReMBResult:
        """
        Return the cached fit result.
        """
        if self._result is None:
            raise RuntimeError("Model is not calibrated. Call fit() first.")
        return self._result

    def probabilities(self) -> Array:
        """
        Return the cached binary-probability matrix.
        """
        return self.result().P

    def expected_weights(self) -> Array:
        """
        Return the cached unconditional expected-weight matrix.
        """
        return self.result().expected_weights


__all__ = [
    "fit_z_from_strengths",
    "fit_delta_from_strengths",
    "probability_matrix_from_strengths",
    "expected_degrees_from_strengths",
    "expected_degrees_firm",
    "expected_weights_matrix_from_strengths",
    "conditional_expected_weights_matrix_from_strengths",
    "rho_from_total_weight",
    "CReMBResult",
    "CReMBModel",
]
