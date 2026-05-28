# Synthetic ITN-like dataset

This folder contains a small anonymized, synthetic weighted network used by the notebook and automated tests.

The file `synthetic_itn_like.npz` contains:

- `W_country`, a symmetric country-level weighted matrix with zero diagonal,
- `W_macro`, the macro-regional aggregation of `W_country`, with diagonal entries retained,
- `country_ids`, anonymized country labels,
- `macro_ids`, macro-region labels for each country,
- `macro_names`, anonymized macro-region names,
- `delta_true`, the simulation value used to generate the latent topology.

The dataset is not a redistributed empirical ITN snapshot.
