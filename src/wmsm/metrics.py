"""Evaluation metrics for probabilistic network reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


def support_mask(n: int, *, directed: bool = False, include_diagonal: bool = False) -> Array:
    """Return the Boolean dyadic support used for link-count comparisons."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if directed:
        mask = np.ones((n, n), dtype=bool)
        if not include_diagonal:
            np.fill_diagonal(mask, False)
        return mask
    return np.triu(np.ones((n, n), dtype=bool), k=0 if include_diagonal else 1)


def link_count(A: Array, *, directed: bool = False, include_diagonal: bool = False) -> float:
    """Compute the number of observed links on the selected dyadic support."""
    adjacency = np.asarray(A, dtype=float) > 0.0
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A must be a square matrix.")
    mask = support_mask(adjacency.shape[0], directed=directed, include_diagonal=include_diagonal)
    return float(np.sum(adjacency[mask]))


def degree_sequence(A_or_P: Array, *, directed: bool = False) -> Array | tuple[Array, Array]:
    """Compute degree sequences from an adjacency or probability matrix."""
    matrix = np.asarray(A_or_P, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The input must be a square matrix.")
    if directed:
        return matrix.sum(axis=1), matrix.sum(axis=0)
    return matrix.sum(axis=1)


def degree_error_metrics(k_obs: Array, k_exp: Array) -> dict[str, float]:
    """Return average and maximum relative errors of an expected degree sequence."""
    observed = np.asarray(k_obs, dtype=float).reshape(-1)
    expected = np.asarray(k_exp, dtype=float).reshape(-1)
    if observed.shape != expected.shape:
        raise ValueError("k_obs and k_exp must have the same shape.")

    active = observed > 0.0
    if not np.any(active):
        raise ValueError("At least one observed degree must be positive.")

    rel = np.abs(expected[active] / observed[active] - 1.0)
    return {
        "ARE_k": float(np.mean(rel)),
        "MRE_k": float(np.max(rel)),
    }


def strength_error_metrics(s_obs: Array, s_exp: Array) -> dict[str, float]:
    """Return average and maximum relative errors of an expected strength sequence."""
    observed = np.asarray(s_obs, dtype=float).reshape(-1)
    expected = np.asarray(s_exp, dtype=float).reshape(-1)
    if observed.shape != expected.shape:
        raise ValueError("s_obs and s_exp must have the same shape.")

    active = observed > 0.0
    if not np.any(active):
        raise ValueError("At least one observed strength must be positive.")

    rel = np.abs(expected[active] / observed[active] - 1.0)
    return {
        "ARE_s": float(np.mean(rel)),
        "MRE_s": float(np.max(rel)),
    }


def binary_classification_metrics(
    A: Array,
    P: Array,
    *,
    directed: bool = False,
    include_diagonal: bool = False,
) -> dict[str, float]:
    """Compute expected confusion-matrix scores from observed links and probabilities."""
    adjacency = (np.asarray(A, dtype=float) > 0.0).astype(float)
    probabilities = np.asarray(P, dtype=float)
    if adjacency.shape != probabilities.shape or adjacency.ndim != 2:
        raise ValueError("A and P must be square matrices with the same shape.")
    if np.any((probabilities < -1e-12) | (probabilities > 1.0 + 1e-12)):
        raise ValueError("P must contain probabilities in the interval [0, 1].")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    mask = support_mask(adjacency.shape[0], directed=directed, include_diagonal=include_diagonal)
    a = adjacency[mask]
    p = probabilities[mask]

    tp = float(np.sum(a * p))
    fp = float(np.sum((1.0 - a) * p))
    tn = float(np.sum((1.0 - a) * (1.0 - p)))
    fn = float(np.sum(a * (1.0 - p)))

    tpr_den = tp + fn
    ppv_den = tp + fp
    tnr_den = tn + fp
    acc_den = tp + fp + tn + fn

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": float(tp / tpr_den) if tpr_den > 0.0 else float("nan"),
        "PPV": float(tp / ppv_den) if ppv_den > 0.0 else float("nan"),
        "TNR": float(tn / tnr_den) if tnr_den > 0.0 else float("nan"),
        "ACC": float((tp + tn) / acc_den) if acc_den > 0.0 else float("nan"),
    }


def reconstruction_summary(
    A: Array,
    W: Array,
    P: Array,
    W_exp: Array,
    *,
    directed: bool = False,
    include_diagonal: bool = False,
) -> dict[str, float]:
    """Compute the scalar diagnostics used in the wMSM empirical comparison."""
    adjacency = (np.asarray(A, dtype=float) > 0.0).astype(float)
    weights = np.asarray(W, dtype=float)
    probabilities = np.asarray(P, dtype=float)
    expected_weights = np.asarray(W_exp, dtype=float)

    if not (adjacency.shape == weights.shape == probabilities.shape == expected_weights.shape):
        raise ValueError("A, W, P, and W_exp must have the same shape.")

    mask = support_mask(adjacency.shape[0], directed=directed, include_diagonal=include_diagonal)
    L_obs = float(np.sum(adjacency[mask]))
    L_exp = float(np.sum(probabilities[mask]))
    rel_L = abs(L_exp - L_obs) / L_obs if L_obs > 0.0 else float("nan")

    if directed:
        k_out_obs, k_in_obs = degree_sequence(adjacency, directed=True)
        k_out_exp, k_in_exp = degree_sequence(probabilities, directed=True)
        k_metrics = degree_error_metrics(
            np.concatenate([k_out_obs, k_in_obs]),
            np.concatenate([k_out_exp, k_in_exp]),
        )
        s_obs = np.concatenate([weights.sum(axis=1), weights.sum(axis=0)])
        s_exp = np.concatenate([expected_weights.sum(axis=1), expected_weights.sum(axis=0)])
    else:
        k_metrics = degree_error_metrics(
            degree_sequence(adjacency, directed=False),
            degree_sequence(probabilities, directed=False),
        )
        s_obs = weights.sum(axis=1)
        s_exp = expected_weights.sum(axis=1)

    out: dict[str, Any] = {
        "L_obs": L_obs,
        "L_exp": L_exp,
        "RE_L": float(rel_L),
    }
    out.update(k_metrics)
    out.update(strength_error_metrics(s_obs, s_exp))
    out.update(
        binary_classification_metrics(
            adjacency,
            probabilities,
            directed=directed,
            include_diagonal=include_diagonal,
        )
    )
    return {key: float(value) for key, value in out.items()}
