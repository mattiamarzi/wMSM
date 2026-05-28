# Mathematical reference

This page collects the equations implemented in `wmsm`.

## Binary MultiScale Model

For an undirected dyad with node strengths $s_i,s_j$, the binary MultiScale Model probability is

$$
p_{ij}=1-\exp(-\delta s_i s_j).
$$

For a partition into blocks $I,J$ with additive strengths $s_I=\sum_{i\in I}s_i$, the same form is preserved:

$$
p_{IJ}=1-\exp(-\delta s_I s_J).
$$

This closure is the reason why a value of $\delta$ calibrated on a coarse layer can be transferred to a finer layer.

## Weighted MultiScale Model

wMSM defines an integer-valued dyadic weight as a compound-Poisson random variable,

$$
W_{ij}=\sum_{r=1}^{K_{ij}}X_{ij,r},
\qquad
K_{ij}\sim\mathrm{Poisson}(\lambda_{ij}),
$$

with

$$
\lambda_{ij}=\delta s_i s_j.
$$

The marks are geometric on positive integers,

$$
P(X=n)=\rho(1-\rho)^{n-1},
\qquad n\ge 1.
$$

The probability of absence and the binary projection are

$$
P(W_{ij}=0)=\exp(-\delta s_i s_j),
\qquad
P(W_{ij}>0)=1-\exp(-\delta s_i s_j).
$$

The expected weight is

$$
E[W_{ij}]=\frac{\delta}{\rho}s_i s_j.
$$

If the full dyadic support is retained, choosing

$$
\rho=\delta W^*,
\qquad
W^*=\sum_i s_i,
$$

gives $E[s_i]=s_i$. If diagonal dyads are removed after constructing the gravity expectation, the remaining expected strengths include the corresponding off-diagonal correction.

## Calibration

At a calibration layer, `fit_delta_from_strengths` solves

$$
\sum_{(i,j)\in\mathcal D}\left[1-\exp(-\delta s_i s_j)\right]=L,
$$

where $\mathcal D$ is the selected dyadic support and $L$ is the observed number of links on that support.

## CReMB benchmark

CReMB uses the density-corrected Gravity Model for the binary layer,

$$
p_{ij}=\frac{z s_i s_j}{1+z s_i s_j},
$$

where $z$ is calibrated by matching the target-layer link count. Its unconditional expected weights are

$$
E[W_{ij}]=\frac{s_i s_j}{W^*}.
$$

The conditional mean assigned to positive links is therefore

$$
E[W_{ij}\mid A_{ij}=1]=\frac{s_i s_j}{W^*p_{ij}}.
$$

## Evaluation metrics

The expected confusion-matrix entries are

$$
\mathrm{TP}=\sum_{(i,j)\in\mathcal D}a_{ij}p_{ij},
\qquad
\mathrm{FP}=\sum_{(i,j)\in\mathcal D}(1-a_{ij})p_{ij},
$$

$$
\mathrm{TN}=\sum_{(i,j)\in\mathcal D}(1-a_{ij})(1-p_{ij}),
\qquad
\mathrm{FN}=\sum_{(i,j)\in\mathcal D}a_{ij}(1-p_{ij}).
$$

The package reports

$$
\mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}},
\quad
\mathrm{PPV}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},
\quad
\mathrm{TNR}=\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}},
\quad
\mathrm{ACC}=\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{FP}+\mathrm{TN}+\mathrm{FN}}.
$$

Degree and strength errors are reported as average and maximum relative errors.
