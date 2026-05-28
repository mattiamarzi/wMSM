# wMSM

`wmsm` is a research package for reconstructing weighted networks across aggregation levels with the **weighted MultiScale Model** (wMSM).

The package is designed for settings in which the binary density is observable only at a coarse scale, while node strengths are available at the finer scale where the network must be reconstructed. The key idea is to calibrate the binary parameter on the aggregate layer and transfer it unchanged to the target layer, because the wMSM probability law is closed under coarse graining with additive strengths.

The repository also includes CReMB, a fine-scale benchmark based on the density-corrected Gravity Model topology and the Conditional Reconstruction Method for weights.

## Installation

```bash
pip install wmsm
```

For development:

```bash
git clone https://github.com/mattiamarzi/wmsm
cd wmsm
pip install -e ".[dev,notebooks]"
pytest -q
```

## Quick start

```python
import numpy as np

from wmsm import load_synthetic_itn_like, link_count
from wmsm import wmsm, cremb
from wmsm.metrics import reconstruction_summary

# Bundled anonymized ITN-like example.
data = load_synthetic_itn_like()
W_country = data["W_country"]
W_macro = data["W_macro"]

A_country = (W_country > 0).astype(float)
A_macro = (W_macro > 0).astype(float)

s_country = W_country.sum(axis=1)
s_macro = W_macro.sum(axis=1)
W_total = W_country.sum()

# wMSM, calibrated on macro-regions and transferred to countries.
L_macro = link_count(A_macro, directed=False, include_diagonal=True)
delta = wmsm.fit_delta_from_strengths(
    s_macro,
    None,
    L_macro,
    directed=False,
    include_diagonal=True,
)
rho = wmsm.rho_from_total_weight(delta, W_total)
P_wmsm = wmsm.probability_matrix_from_strengths(
    s_country,
    None,
    delta=delta,
    directed=False,
    include_diagonal=False,
)
Wexp_wmsm = wmsm.expected_weights_matrix_from_strengths(
    s_country,
    None,
    delta=delta,
    rho=rho,
    directed=False,
    include_diagonal=False,
)

# CReMB, calibrated directly on the country layer.
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

summary_wmsm = reconstruction_summary(
    A_country,
    W_country,
    P_wmsm,
    Wexp_wmsm,
    directed=False,
    include_diagonal=False,
)
summary_cremb = reconstruction_summary(
    A_country,
    W_country,
    P_cremb,
    Wexp_cremb,
    directed=False,
    include_diagonal=False,
)

print(summary_wmsm["ARE_k"], summary_wmsm["MRE_k"], summary_wmsm["TPR"])
print(summary_cremb["ARE_k"], summary_cremb["MRE_k"], summary_cremb["TPR"])
```

## Repository layout

```text
src/wmsm/wmsm.py      wMSM calibration and first moments
src/wmsm/cremb.py      CReMB benchmark calibration and first moments
src/wmsm/metrics.py    diagnostics used in the paper
datasets/               anonymized synthetic example dataset
notebooks/              end-to-end reproducible examples
docs/                   usage notes, API reference, and equations
tests/                  automated tests for the package workflow
```

## Documentation

See:

- `docs/usage.md`
- `docs/api_quick_reference.md`
- `docs/math.md`

## Citation

The companion paper is currently in preparation. Until the public bibliographic record is available, please cite the repository and use the placeholder `CITATION.cff` file.

## License

MIT, see `LICENSE`.
