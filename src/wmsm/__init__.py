"""Multiscale reconstruction of weighted networks across aggregation levels."""

from .wmsm import (
    WMSMModel,
    WMSMResult,
    expected_degrees_from_strengths,
    expected_weights_matrix_from_strengths,
    fit_delta_from_strengths,
    probability_matrix_from_strengths,
    rho_from_total_weight,
)
from .cremb import CReMBModel, CReMBResult, fit_z_from_strengths
from .datasets import aggregate_by_labels, load_synthetic_itn_like
from .metrics import (
    binary_classification_metrics,
    degree_error_metrics,
    link_count,
    reconstruction_summary,
    strength_error_metrics,
    support_mask,
)

__all__ = [
    "WMSMModel",
    "WMSMResult",
    "CReMBModel",
    "CReMBResult",
    "aggregate_by_labels",
    "binary_classification_metrics",
    "degree_error_metrics",
    "expected_degrees_from_strengths",
    "expected_weights_matrix_from_strengths",
    "fit_delta_from_strengths",
    "fit_z_from_strengths",
    "link_count",
    "load_synthetic_itn_like",
    "probability_matrix_from_strengths",
    "reconstruction_summary",
    "rho_from_total_weight",
    "strength_error_metrics",
    "support_mask",
]

__version__ = "0.1.0"
