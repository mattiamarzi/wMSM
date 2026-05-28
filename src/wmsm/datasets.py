"""Dataset helpers used by examples, notebooks, and tests."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray


def _load_npz_from_package(filename: str) -> Any:
    """Open a compressed NumPy dataset bundled with the installed package."""
    try:
        package_files = resources.files("wmsm.data")
        with resources.as_file(package_files / filename) as path:
            return np.load(path, allow_pickle=False)
    except (FileNotFoundError, ModuleNotFoundError):
        root = Path(__file__).resolve().parents[2]
        fallback = root / "datasets" / "synthetic_itn_like" / filename
        return np.load(fallback, allow_pickle=False)


def load_synthetic_itn_like() -> dict[str, Array]:
    """Return the anonymized ITN-like weighted network bundled with the package.

    The dataset contains a country-level undirected weighted matrix, a
    macro-regional partition, and the corresponding macro-regional matrix. It
    is generated synthetically from a heterogeneous multiscale topology and
    gravity-like positive weights, and it is not a redistributed empirical ITN
    snapshot.
    """
    with _load_npz_from_package("synthetic_itn_like.npz") as data:
        return {
            "W_country": data["W_country"].astype(float),
            "W_macro": data["W_macro"].astype(float),
            "country_ids": data["country_ids"].astype(str),
            "macro_ids": data["macro_ids"].astype(int),
            "macro_names": data["macro_names"].astype(str),
            "delta_true": data["delta_true"].astype(float),
        }


def aggregate_by_labels(W: Array, labels: Array) -> Array:
    """Aggregate a weighted matrix by summing entries inside label blocks.

    The operation preserves row-strength additivity. If the input matrix is
    symmetric, diagonal block entries of the output contain the total weight
    internal to each block under the same matrix convention as the input.
    """
    matrix = np.asarray(W, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("W must be a square matrix.")

    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1 or labels_arr.size != matrix.shape[0]:
        raise ValueError("labels must be a one-dimensional array aligned with W.")

    unique_labels = np.unique(labels_arr)
    out = np.zeros((unique_labels.size, unique_labels.size), dtype=float)
    for a, label_a in enumerate(unique_labels):
        idx_a = np.where(labels_arr == label_a)[0]
        for b, label_b in enumerate(unique_labels):
            idx_b = np.where(labels_arr == label_b)[0]
            out[a, b] = float(matrix[np.ix_(idx_a, idx_b)].sum())
    return out
