#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted MultiScale Model (wMSM).

Model
-----
For every ordered pair of nodes ``(i, j)``,

    K_ij ~ Poisson(lambda_ij),
    W_ij = sum_{r=1}^{K_ij} X_{ij,r},

where the marks ``X_ij,r`` are i.i.d. geometric on ``{1, 2, ...}`` with
parameter ``rho``,

    P(X = n) = rho (1 - rho)^{n-1},    n >= 1,

and the intensity is

    lambda_ij = delta * s_out[i] * s_in[j].

The induced binary projection is

    p_ij = P(W_ij > 0) = 1 - exp(-delta * s_out[i] * s_in[j]),

while the expected dyadic weight is

    E[W_ij] = (delta / rho) * s_out[i] * s_in[j].

Aggregation
-----------
Under coarse graining, strengths add exactly. Therefore the aggregated model
keeps the same functional form after replacing micro strengths with aggregated
strengths.

Implementation notes
--------------------
- Group-level routines operate on dense outer products, which is entirely
  adequate for the scales considered in the current workflows.
- Firm-level expected degrees are evaluated in a streaming fashion to avoid
  allocating a dense ``N x N`` matrix.
- Self-loops can be included or excluded through the ``include_diagonal`` flag.
  The default is ``True`` for backward compatibility with the original
  production-network notebook.
- The generic array-based API supports both directed and undirected networks
  through the ``directed`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from numba import njit

Array = np.ndarray


# =============================================================================
# Numba-compiled primitives
# =============================================================================
@njit(cache=True, fastmath=True)
def _expected_L_and_grad(delta: float, M: Array) -> Tuple[float, float]:
    """
    Compute ``E[L(delta)]`` and its derivative for ``P = 1 - exp(-delta * M)``.

    Parameters
    ----------
    delta:
        Sparsity parameter.
    M:
        Nonnegative dyadic scale matrix.
    """
    EL = 0.0
    dEL = 0.0
    n0, n1 = M.shape
    for i in range(n0):
        for j in range(n1):
            mij = M[i, j]
            if mij <= 0.0:
                continue
            e = np.exp(-delta * mij)
            EL += 1.0 - e
            dEL += mij * e
    return EL, dEL


@njit(cache=True, fastmath=True)
def _degrees_from_M(delta: float, M: Array) -> Tuple[Array, Array, float]:
    """
    Compute expected out-degrees, in-degrees, and expected link count.

    The matrix ``M`` is interpreted through ``P = 1 - exp(-delta * M)``.
    """
    n0, n1 = M.shape
    kout = np.zeros(n0, dtype=np.float64)
    kin = np.zeros(n1, dtype=np.float64)
    EL = 0.0

    for i in range(n0):
        row_sum = 0.0
        for j in range(n1):
            mij = M[i, j]
            if mij <= 0.0:
                continue
            pij = 1.0 - np.exp(-delta * mij)
            row_sum += pij
            kin[j] += pij
            EL += pij
        kout[i] = row_sum

    return kout, kin, EL


@njit(cache=True, fastmath=True)
def _expected_degrees_firm_streaming(
    sout: Array,
    sin: Array,
    delta: float,
    chunk_size: int,
    include_diagonal: bool,
) -> Tuple[Array, Array, float]:
    """
    Streaming computation of expected degrees without allocating a dense matrix.
    """
    n = sout.shape[0]
    kout = np.zeros(n, dtype=np.float64)
    kin = np.zeros(n, dtype=np.float64)
    EL = 0.0

    for j0 in range(0, n, chunk_size):
        j1 = j0 + chunk_size
        if j1 > n:
            j1 = n

        for i in range(n):
            si = sout[i]
            if si <= 0.0:
                continue
            row_acc = 0.0
            for j in range(j0, j1):
                if (not include_diagonal) and (i == j):
                    continue
                sj = sin[j]
                if sj <= 0.0:
                    continue
                mij = si * sj
                pij = 1.0 - np.exp(-delta * mij)
                row_acc += pij
                kin[j] += pij
                EL += pij
            kout[i] += row_acc

    return kout, kin, EL


# =============================================================================
# Internal helpers
# =============================================================================
def _norm_str_series(x: pd.Series) -> pd.Series:
    """Return a stripped pandas string series suitable for identifier matching."""
    return x.astype("string").str.strip()


def _apply_diagonal_policy(M: Array, include_diagonal: bool) -> Array:
    """
    Return a copy of ``M`` with the diagonal removed when requested.

    For rectangular matrices the input is returned unchanged.
    """
    M = np.asarray(M, dtype=np.float64)
    if include_diagonal or M.shape[0] != M.shape[1]:
        return M
    M2 = M.copy()
    np.fill_diagonal(M2, 0.0)
    return M2


def _validate_undirected_strengths(
    sout: Array,
    sin: Optional[Array],
    *,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> Array:
    """Return the unique strength vector for an undirected network."""
    s = np.asarray(sout, dtype=np.float64).ravel()
    if sin is None:
        return s
    t = np.asarray(sin, dtype=np.float64).ravel()
    if s.shape != t.shape:
        raise ValueError("sout and sin must have the same length.")
    if not np.allclose(s, t, atol=atol, rtol=rtol):
        raise ValueError(
            "Undirected mode requires a single strength sequence. "
            "The provided out- and in-strength vectors are not equal."
        )
    return s


def _undirected_upper_triangle_mask(n: int, include_diagonal: bool) -> Array:
    """Return the mask selecting the undirected dyadic support."""
    return np.triu(np.ones((n, n), dtype=bool), k=0 if include_diagonal else 1)


def _expected_L_and_grad_undirected(
    delta: float,
    s: Array,
    *,
    include_diagonal: bool,
) -> Tuple[float, float]:
    """Compute ``E[L(delta)]`` and its derivative in undirected mode."""
    M = np.outer(s, s)
    if not include_diagonal:
        np.fill_diagonal(M, 0.0)
    mask = _undirected_upper_triangle_mask(M.shape[0], include_diagonal=include_diagonal)
    Mv = M[mask]
    if Mv.size == 0:
        return 0.0, 0.0
    positive = Mv > 0.0
    if not np.any(positive):
        return 0.0, 0.0
    Mv = Mv[positive]
    e = np.exp(-delta * Mv)
    EL = float(np.sum(1.0 - e))
    dEL = float(np.sum(Mv * e))
    return EL, dEL


def _degrees_from_strengths_undirected(
    s: Array,
    *,
    delta: float,
    include_diagonal: bool,
) -> Tuple[Array, Array, float]:
    """Compute expected degrees and expected link count in undirected mode."""
    M = np.outer(s, s)
    if not include_diagonal:
        np.fill_diagonal(M, 0.0)
    P = 1.0 - np.exp(-float(delta) * M)
    k = np.sum(P, axis=1)
    mask = _undirected_upper_triangle_mask(P.shape[0], include_diagonal=include_diagonal)
    EL = float(np.sum(P[mask]))
    return k, k.copy(), EL


def _aggregate_group_strengths(strengths_group: pd.DataFrame, *, goods: str) -> pd.DataFrame:
    """Aggregate potential duplicate group-strength rows to a unique group index."""
    sg = strengths_group[strengths_group["ggnr"] == goods].copy()
    if sg.empty:
        return sg
    sg["group"] = _norm_str_series(sg["group"])
    sg["s_out"] = pd.to_numeric(sg["s_out"], errors="coerce").fillna(0.0)
    sg["s_in"] = pd.to_numeric(sg["s_in"], errors="coerce").fillna(0.0)
    sg = sg.groupby("group", as_index=False)[["s_out", "s_in"]].sum()
    sg["ggnr"] = goods
    return sg


def _infer_groups_from_io_and_strengths(
    io_flows: pd.DataFrame,
    strengths_group: pd.DataFrame,
    *,
    goods: str,
) -> List[str]:
    """Infer a consistent group ordering using the common IO/strength support."""
    dfW = io_flows[io_flows["ggnr"] == goods].copy()
    if dfW.empty:
        raise ValueError(f"No IO flows found for goods={goods!r}.")
    dfW["r"] = _norm_str_series(dfW["r"])
    dfW["s"] = _norm_str_series(dfW["s"])
    io_groups = set(dfW["r"].tolist()) | set(dfW["s"].tolist())

    sg = _aggregate_group_strengths(strengths_group, goods=goods)
    if sg.empty:
        raise ValueError(f"No group strengths found for goods={goods!r}.")
    st_groups = set(sg["group"].tolist())

    common = sorted(list(io_groups & st_groups))
    if len(common) == 0:
        raise ValueError("No common group labels between IO flows and group strengths.")
    return common


def _strength_vectors_from_group_frame(
    strengths_group: pd.DataFrame,
    *,
    goods: str,
    groups: Optional[List[str]],
) -> Tuple[List[str], Array, Array]:
    """
    Extract aligned out- and in-strength vectors from the group-strength frame.
    """
    if not {"group", "ggnr", "s_out", "s_in"}.issubset(strengths_group.columns):
        raise ValueError("strengths_group must contain columns: group, ggnr, s_out, s_in")

    sg = _aggregate_group_strengths(strengths_group, goods=goods)
    if sg.empty:
        raise ValueError(f"No group strengths found for goods={goods!r}.")

    if groups is None:
        groups = sorted(sg["group"].unique().tolist())
    else:
        groups = [str(g).strip() for g in groups if str(g).strip() != ""]
        if len(groups) == 0:
            raise ValueError("groups must be a non-empty list when provided.")

    gset: Set[str] = set(groups)
    sg = sg[sg["group"].isin(gset)].copy()

    sout = sg.set_index("group")["s_out"].reindex(groups).fillna(0.0).to_numpy(dtype=np.float64)
    sin = sg.set_index("group")["s_in"].reindex(groups).fillna(0.0).to_numpy(dtype=np.float64)
    return groups, sout, sin


# =============================================================================
# Generic array-based public API
# =============================================================================
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
    Fit ``delta`` by matching the observed number of binary links.

    Parameters
    ----------
    sout, sin:
        Out- and in-strength vectors in directed mode. In undirected mode,
        ``sin`` can be omitted or must coincide with ``sout``.
    L_obs:
        Observed number of links at the corresponding scale.
    include_diagonal:
        Whether diagonal dyads are allowed.
    directed:
        Whether the network is directed.
    tol:
        Relative tolerance on the link-count matching equation.
    max_iter:
        Maximum number of safeguarded Newton iterations.
    """
    sout = np.asarray(sout, dtype=np.float64).ravel()
    if directed:
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        sin = np.asarray(sin, dtype=np.float64).ravel()
        if sout.ndim != 1 or sin.ndim != 1:
            raise ValueError("sout and sin must be one-dimensional arrays.")
        if sout.size != sin.size:
            raise ValueError("sout and sin must have the same length.")
    else:
        sout = _validate_undirected_strengths(sout, sin)
        sin = None

    if L_obs <= 0.0:
        return 0.0

    if directed:
        M = _apply_diagonal_policy(np.outer(sout, sin), include_diagonal=include_diagonal)
        n_pairs = int(M.size if include_diagonal else (M.size - min(M.shape)))
        expected_fn = lambda value: _expected_L_and_grad(value, M)
    else:
        n = int(sout.size)
        n_pairs = int(n * (n + 1) // 2) if include_diagonal else int(n * (n - 1) // 2)
        expected_fn = lambda value: _expected_L_and_grad_undirected(
            value,
            sout,
            include_diagonal=include_diagonal,
        )

    lo = 0.0
    hi = 1.0
    EL_hi, _ = expected_fn(hi)
    while EL_hi < L_obs and EL_hi < 0.999 * n_pairs:
        hi *= 2.0
        if hi > 1e12:
            break
        EL_hi, _ = expected_fn(hi)

    if EL_hi < L_obs:
        return float(hi)

    delta = 0.5 * (lo + hi)
    for _ in range(max_iter):
        EL, dEL = expected_fn(delta)
        err = EL - float(L_obs)
        if abs(err) / max(float(L_obs), 1.0) < tol:
            return float(delta)

        cand = delta - err / dEL if dEL > 0.0 else 0.5 * (lo + hi)
        if not (lo < cand < hi):
            cand = 0.5 * (lo + hi)

        if err < 0.0:
            lo = delta
        else:
            hi = delta

        delta = cand

    return float(delta)

def probability_matrix_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Array:
    """
    Compute the dense link-probability matrix from strength vectors.
    """
    sout = np.asarray(sout, dtype=np.float64).ravel()
    if directed:
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        sin = np.asarray(sin, dtype=np.float64).ravel()
        M = _apply_diagonal_policy(np.outer(sout, sin), include_diagonal=include_diagonal)
    else:
        s = _validate_undirected_strengths(sout, sin)
        M = np.outer(s, s)
        M = _apply_diagonal_policy(M, include_diagonal=include_diagonal)
    return 1.0 - np.exp(-float(delta) * M)

def expected_degrees_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Tuple[Array, Array, float]:
    """
    Compute expected degrees and expected link count from strength vectors.
    """
    sout = np.asarray(sout, dtype=np.float64).ravel()
    if directed:
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        sin = np.asarray(sin, dtype=np.float64).ravel()
        M = _apply_diagonal_policy(np.outer(sout, sin), include_diagonal=include_diagonal)
        kout, kin, EL = _degrees_from_M(float(delta), M)
        return kout, kin, float(EL)

    s = _validate_undirected_strengths(sout, sin)
    kout, kin, EL = _degrees_from_strengths_undirected(
        s,
        delta=float(delta),
        include_diagonal=include_diagonal,
    )
    return kout, kin, float(EL)

def expected_weights_matrix_from_strengths(
    sout: Array,
    sin: Optional[Array] = None,
    *,
    delta: float,
    rho: float,
    include_diagonal: bool = True,
    directed: bool = True,
) -> Array:
    """
    Compute the dense expected-weight matrix from strength vectors.
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    sout = np.asarray(sout, dtype=np.float64).ravel()
    if directed:
        if sin is None:
            raise ValueError("sin must be provided in directed mode.")
        sin = np.asarray(sin, dtype=np.float64).ravel()
        Wexp = (float(delta) / float(rho)) * np.outer(sout, sin)
    else:
        s = _validate_undirected_strengths(sout, sin)
        Wexp = (float(delta) / float(rho)) * np.outer(s, s)
    Wexp = _apply_diagonal_policy(Wexp, include_diagonal=include_diagonal)
    return Wexp

def fit_delta_group(
    io_flows: pd.DataFrame,
    strengths_group: pd.DataFrame,
    *,
    goods: str,
    tol: float = 1e-8,
    max_iter: int = 200,
    groups: Optional[List[str]] = None,
    include_diagonal: bool = True,
) -> float:
    """
    Fit ``delta`` by matching the observed number of directed links at group scale.
    """
    if not {"ggnr", "r", "s", "W"}.issubset(io_flows.columns):
        raise ValueError("io_flows must contain columns: ggnr, r, s, W")
    if goods is None:
        raise ValueError("goods must be a single label.")

    if groups is None:
        groups = _infer_groups_from_io_and_strengths(io_flows, strengths_group, goods=goods)
    else:
        groups = [str(g).strip() for g in groups if str(g).strip() != ""]
        if len(groups) == 0:
            raise ValueError("groups must be a non-empty list when provided.")

    gset: Set[str] = set(groups)

    dfW = io_flows[io_flows["ggnr"] == goods].copy()
    dfW = dfW.groupby(["r", "s"], as_index=False)["W"].sum()
    dfW["r"] = _norm_str_series(dfW["r"])
    dfW["s"] = _norm_str_series(dfW["s"])
    dfW = dfW[dfW["r"].isin(gset) & dfW["s"].isin(gset)].copy()
    if dfW.empty:
        raise ValueError("IO flow groups do not overlap with the requested group set.")

    A = (pd.to_numeric(dfW["W"], errors="coerce").fillna(0.0) > 0.0).astype(int)
    if not include_diagonal:
        A = A[dfW["r"] != dfW["s"]]
    L_obs = int(A.sum())
    if L_obs == 0:
        return 0.0

    _, sout, sin = _strength_vectors_from_group_frame(strengths_group, goods=goods, groups=groups)
    return fit_delta_from_strengths(
        sout,
        sin,
        L_obs,
        include_diagonal=include_diagonal,
        tol=tol,
        max_iter=max_iter,
    )


def expected_degrees_group(
    strengths_group: pd.DataFrame,
    *,
    goods: str,
    delta: float,
    groups: Optional[List[str]] = None,
    include_diagonal: bool = True,
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Compute expected out-degrees, in-degrees, and link count at group level.
    """
    if goods is None:
        raise ValueError("goods must be a single label.")
    groups, sout, sin = _strength_vectors_from_group_frame(strengths_group, goods=goods, groups=groups)
    kout, kin, EL = expected_degrees_from_strengths(
        sout,
        sin,
        delta=delta,
        include_diagonal=include_diagonal,
    )
    return pd.Series(kout, index=groups), pd.Series(kin, index=groups), float(EL)


def expected_degrees_firm(
    firms: pd.DataFrame,
    strengths: pd.DataFrame,
    *,
    goods: str,
    delta: float,
    chunk_size: int = 1024,
    include_diagonal: bool = True,
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Compute expected firm-level degrees for a given goods label.

    The implementation is memory-stable and does not allocate a dense matrix.
    """
    if goods is None:
        raise ValueError("expected_degrees_firm requires goods to be a single label.")
    if "beid" not in firms.columns:
        raise ValueError("firms must contain column: beid")
    if not {"beid", "ggnr", "s_out", "s_in"}.issubset(strengths.columns):
        raise ValueError("strengths must contain columns: beid, ggnr, s_out, s_in")

    s = strengths[strengths["ggnr"] == goods].copy()
    if s.empty:
        raise ValueError(f"No strengths found for goods={goods!r}.")

    s["beid"] = _norm_str_series(s["beid"])
    s["s_out"] = pd.to_numeric(s["s_out"], errors="coerce").fillna(0.0)
    s["s_in"] = pd.to_numeric(s["s_in"], errors="coerce").fillna(0.0)
    s = s.groupby("beid", as_index=False)[["s_out", "s_in"]].sum()

    df = firms[["beid"]].copy()
    df["beid"] = _norm_str_series(df["beid"])
    df = df.merge(s, on="beid", how="inner", validate="one_to_one")
    if df.empty:
        raise ValueError("No firms remain after merging strengths.")

    sout = df["s_out"].to_numpy(dtype=np.float64)
    sin = df["s_in"].to_numpy(dtype=np.float64)

    kout, kin, EL = _expected_degrees_firm_streaming(
        sout,
        sin,
        float(delta),
        int(chunk_size),
        bool(include_diagonal),
    )
    beids = df["beid"].astype("string").astype(str).tolist()
    return pd.Series(kout, index=beids), pd.Series(kin, index=beids), float(EL)


def rho_from_total_weight(delta: float, total_weight: float) -> float:
    """
    Return the geometric parameter ``rho = delta * W*`` implied by the moment condition.
    """
    if total_weight <= 0.0:
        raise ValueError("total_weight must be positive.")
    if delta < 0.0:
        raise ValueError("delta must be non-negative.")
    rho = float(delta) * float(total_weight)
    if not (0.0 < rho < 1.0):
        raise ValueError(
            "rho must lie in (0, 1) for the geometric marks used by wMSM. "
            f"Got rho={rho:.6g} from delta={float(delta):.6g} and W*={float(total_weight):.6g}."
        )
    return float(rho)


def expected_weights_group(
    strengths_group: pd.DataFrame,
    *,
    goods: str,
    delta: float,
    rho: float,
    groups: Optional[List[str]] = None,
    include_diagonal: bool = True,
) -> pd.DataFrame:
    """
    Compute the expected dyadic weight matrix at group level.
    """
    if goods is None:
        raise ValueError("goods must be a single label.")
    groups, sout, sin = _strength_vectors_from_group_frame(strengths_group, goods=goods, groups=groups)
    Wexp = expected_weights_matrix_from_strengths(
        sout,
        sin,
        delta=delta,
        rho=rho,
        include_diagonal=include_diagonal,
    )
    return pd.DataFrame(Wexp, index=groups, columns=groups)

@dataclass(frozen=True)
class WMSMResult:
    """
    Container for calibrated wMSM quantities.

    The object stores the binary scale parameter, the geometric mark parameter,
    the fitted probability matrix, and the corresponding first moments. It is a
    lightweight data container and does not perform additional computations.
    """

    delta: float
    rho: float
    P: Array
    expected_out_degrees: Array
    expected_in_degrees: Array
    expected_links: float
    expected_weights: Array
    expected_out_strengths: Array
    expected_in_strengths: Array
    total_weight: float


class WMSMModel:
    """
    Cache-oriented interface for fitting wMSM on one dyadic support.

    The model estimates the binary scale parameter by matching the expected
    number of links on the supplied support. The geometric mark parameter is then
    fixed by the total-weight moment condition. For multiscale reconstruction,
    one may calibrate ``delta`` on a coarse layer and pass the same value to
    ``probability_matrix_from_strengths`` at a finer layer.
    """

    def __init__(
        self,
        sout: Array,
        sin: Optional[Array] = None,
        *,
        L_obs: float,
        total_weight: Optional[float] = None,
        include_diagonal: bool = True,
        directed: bool = True,
    ) -> None:
        self.directed = bool(directed)
        self.include_diagonal = bool(include_diagonal)
        self.L_obs = float(L_obs)
        if not np.isfinite(self.L_obs) or self.L_obs < 0.0:
            raise ValueError("L_obs must be a finite non-negative scalar.")

        if self.directed:
            self.sout = np.asarray(sout, dtype=np.float64).reshape(-1)
            if sin is None:
                raise ValueError("sin must be provided in directed mode.")
            self.sin = np.asarray(sin, dtype=np.float64).reshape(-1)
            if self.sout.size != self.sin.size:
                raise ValueError("sout and sin must have the same length.")
            inferred_total = float(np.sum(self.sout))
        else:
            s = _validate_undirected_strengths(sout, sin)
            self.sout = s
            self.sin = s.copy()
            inferred_total = float(np.sum(s))

        self.total_weight = float(inferred_total if total_weight is None else total_weight)
        if not np.isfinite(self.total_weight) or self.total_weight <= 0.0:
            raise ValueError("total_weight must be a finite positive scalar.")
        self._result: Optional[WMSMResult] = None

    def fit(self, *, tol: float = 1e-8, max_iter: int = 200) -> WMSMResult:
        """Calibrate ``delta`` and cache the fitted first moments."""
        delta = fit_delta_from_strengths(
            self.sout,
            self.sin,
            self.L_obs,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
            tol=tol,
            max_iter=max_iter,
        )
        rho = rho_from_total_weight(delta, self.total_weight)
        P = probability_matrix_from_strengths(
            self.sout,
            self.sin,
            delta=delta,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
        )
        kout, kin, expected_links = expected_degrees_from_strengths(
            self.sout,
            self.sin,
            delta=delta,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
        )
        Wexp = expected_weights_matrix_from_strengths(
            self.sout,
            self.sin,
            delta=delta,
            rho=rho,
            include_diagonal=self.include_diagonal,
            directed=self.directed,
        )
        result = WMSMResult(
            delta=float(delta),
            rho=float(rho),
            P=P,
            expected_out_degrees=kout,
            expected_in_degrees=kin,
            expected_links=float(expected_links),
            expected_weights=Wexp,
            expected_out_strengths=Wexp.sum(axis=1),
            expected_in_strengths=Wexp.sum(axis=0),
            total_weight=float(self.total_weight),
        )
        self._result = result
        return result

    def result(self) -> WMSMResult:
        """Return the cached fit result."""
        if self._result is None:
            raise RuntimeError("Model is not calibrated. Call fit() first.")
        return self._result

    def probabilities(self) -> Array:
        """Return the cached binary-probability matrix."""
        return self.result().P

    def expected_weights(self) -> Array:
        """Return the cached expected-weight matrix."""
        return self.result().expected_weights


__all__ = [
    "fit_delta_from_strengths",
    "probability_matrix_from_strengths",
    "expected_degrees_from_strengths",
    "expected_degrees_firm",
    "expected_weights_matrix_from_strengths",
    "rho_from_total_weight",
    "expected_degrees_group",
    "expected_weights_group",
    "fit_delta_group",
    "WMSMResult",
    "WMSMModel",
]

