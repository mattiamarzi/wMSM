import numpy as np

from wmsm import cremb, wmsm
from wmsm.datasets import aggregate_by_labels, load_synthetic_itn_like
from wmsm.metrics import link_count, reconstruction_summary


def test_dataset_loader_and_aggregation_are_consistent():
    data = load_synthetic_itn_like()
    W_country = data["W_country"]
    W_macro = data["W_macro"]
    labels = data["macro_ids"]

    assert W_country.shape[0] == labels.size
    assert np.allclose(W_country, W_country.T)
    assert np.allclose(np.diag(W_country), 0.0)
    assert np.allclose(W_macro, aggregate_by_labels(W_country, labels))
    assert np.isclose(W_country.sum(), W_macro.sum())


def test_wmsm_macro_calibration_transfers_to_country_layer():
    data = load_synthetic_itn_like()
    W_country = data["W_country"]
    W_macro = data["W_macro"]

    A_macro = (W_macro > 0).astype(float)
    s_macro = W_macro.sum(axis=1)
    s_country = W_country.sum(axis=1)

    L_macro = link_count(A_macro, directed=False, include_diagonal=True)
    delta = wmsm.fit_delta_from_strengths(
        s_macro,
        None,
        L_macro,
        directed=False,
        include_diagonal=True,
    )
    rho = wmsm.rho_from_total_weight(delta, W_country.sum())

    assert np.isfinite(delta) and delta > 0.0
    assert np.isfinite(rho) and 0.0 < rho < 1.0

    P_macro = wmsm.probability_matrix_from_strengths(
        s_macro,
        None,
        delta=delta,
        directed=False,
        include_diagonal=True,
    )
    assert abs(np.triu(P_macro, 0).sum() - L_macro) / L_macro < 1e-8

    P_country = wmsm.probability_matrix_from_strengths(
        s_country,
        None,
        delta=delta,
        directed=False,
        include_diagonal=False,
    )
    Wexp_country = wmsm.expected_weights_matrix_from_strengths(
        s_country,
        None,
        delta=delta,
        rho=rho,
        directed=False,
        include_diagonal=False,
    )
    summary = reconstruction_summary(
        (W_country > 0).astype(float),
        W_country,
        P_country,
        Wexp_country,
        directed=False,
        include_diagonal=False,
    )

    for key in ["RE_L", "ARE_k", "MRE_k", "TPR", "PPV", "TNR", "ACC"]:
        assert np.isfinite(summary[key])


def test_cremb_matches_target_link_count_on_country_layer():
    data = load_synthetic_itn_like()
    W_country = data["W_country"]
    A_country = (W_country > 0).astype(float)
    s_country = W_country.sum(axis=1)
    L_country = link_count(A_country, directed=False, include_diagonal=False)

    z = cremb.fit_delta_from_strengths(
        s_country,
        None,
        L_country,
        directed=False,
        include_diagonal=False,
    )
    P = cremb.probability_matrix_from_strengths(
        s_country,
        None,
        delta=z,
        directed=False,
        include_diagonal=False,
    )
    assert np.isfinite(z) and z > 0.0
    assert abs(np.triu(P, 1).sum() - L_country) / L_country < 1e-8


def test_container_interfaces_return_cached_moments():
    data = load_synthetic_itn_like()
    W_country = data["W_country"]
    s_country = W_country.sum(axis=1)
    L_country = link_count(W_country > 0, directed=False, include_diagonal=False)

    model = cremb.CReMBModel(
        s_country,
        L_obs=L_country,
        directed=False,
        include_diagonal=False,
    )
    res = model.fit()
    assert res.P.shape == W_country.shape
    assert np.allclose(model.probabilities(), res.P)
    assert np.allclose(model.expected_weights(), res.expected_weights)
