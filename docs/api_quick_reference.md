# API quick reference

## Main modules

```python
from wmsm import wmsm, cremb
from wmsm.datasets import load_synthetic_itn_like, aggregate_by_labels
from wmsm.metrics import reconstruction_summary, binary_classification_metrics
```

## wMSM

```python
wmsm.fit_delta_from_strengths(sout, sin, L_obs, directed=True, include_diagonal=True)
```

Fits the binary parameter $\delta$ by matching the expected number of links.

```python
wmsm.probability_matrix_from_strengths(sout, sin=None, delta=..., directed=False)
```

Builds the dense matrix $P_{ij}=1-\exp(-\delta s_i s_j)$.

```python
wmsm.rho_from_total_weight(delta, total_weight)
```

Returns $\rho=\delta W^*$ and verifies that it lies in $(0,1)$.

```python
wmsm.expected_weights_matrix_from_strengths(sout, sin=None, delta=..., rho=..., directed=False)
```

Builds the dense expected-weight matrix.

```python
model = wmsm.WMSMModel(strengths, L_obs=L, directed=False, include_diagonal=False)
result = model.fit()
```

Small container API for one-scale calibration.

## CReMB

```python
cremb.fit_z_from_strengths(sout, sin, L_obs, directed=True, include_diagonal=True)
cremb.fit_delta_from_strengths(sout, sin, L_obs, directed=True, include_diagonal=True)
```

Both fit the CReMB binary scale. The second function is a compatibility wrapper and returns $z$.

```python
cremb.probability_matrix_from_strengths(sout, sin=None, delta=z, directed=False)
```

Builds the dcGM probability matrix.

```python
cremb.expected_weights_matrix_from_strengths(sout, sin=None, directed=False)
```

Builds the unconditional gravity expectation.

```python
model = cremb.CReMBModel(strengths, L_obs=L, directed=False, include_diagonal=False)
result = model.fit()
```

Container API for one-scale CReMB calibration.

## Metrics

```python
reconstruction_summary(A, W, P, W_exp, directed=False, include_diagonal=False)
```

Returns `RE_L`, `ARE_k`, `MRE_k`, `ARE_s`, `MRE_s`, `TPR`, `PPV`, `TNR`, `ACC`, and the expected confusion-matrix entries.

```python
binary_classification_metrics(A, P, directed=False, include_diagonal=False)
```

Returns expected binary support diagnostics.

```python
link_count(A, directed=False, include_diagonal=False)
```

Counts links on the selected dyadic support.
