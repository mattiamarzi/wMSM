# Usage guide

## Multiscale reconstruction workflow

The typical workflow has two layers:

1. a calibration layer, where a coarse adjacency is observed,
2. a target layer, where strengths are known and the microscopic support is reconstructed.

For an undirected ITN-like system, macro-regional self-loops are retained during calibration because they represent internal block weight, while country-level self-loops are removed during reconstruction.

```python
from wmsm import wmsm, link_count

L_macro = link_count(A_macro, directed=False, include_diagonal=True)
delta = wmsm.fit_delta_from_strengths(
    s_macro,
    None,
    L_macro,
    directed=False,
    include_diagonal=True,
)

rho = wmsm.rho_from_total_weight(delta, W_country.sum())
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
```

## Fine-scale CReMB benchmark

CReMB is calibrated at the same scale at which it is evaluated.

```python
from wmsm import cremb, link_count

L_country = link_count(A_country, directed=False, include_diagonal=False)
z = cremb.fit_delta_from_strengths(
    s_country,
    None,
    L_country,
    directed=False,
    include_diagonal=False,
)
P_cremb = cremb.probability_matrix_from_strengths(
    s_country,
    None,
    delta=z,
    directed=False,
    include_diagonal=False,
)
Wexp_cremb = cremb.expected_weights_matrix_from_strengths(
    s_country,
    None,
    directed=False,
    include_diagonal=False,
)
```

## Diagnostics

```python
from wmsm.metrics import reconstruction_summary

summary = reconstruction_summary(
    A_country,
    W_country,
    P_country,
    Wexp_country,
    directed=False,
    include_diagonal=False,
)

for key in ["RE_L", "ARE_k", "MRE_k", "TPR", "PPV", "TNR", "ACC"]:
    print(key, summary[key])
```

## Bundled synthetic data

The repository ships an anonymized synthetic ITN-like dataset used by the notebook and the tests.

```python
from wmsm.datasets import load_synthetic_itn_like

data = load_synthetic_itn_like()
W_country = data["W_country"]
W_macro = data["W_macro"]
```

The dataset is small, deterministic, and does not contain raw empirical trade data.
