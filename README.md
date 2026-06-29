# IntervalGP-VAE TMLR Revision Experiments

This repository contains the experimental notebooks used for the TMLR revision of **IntervalGP-VAE**, an uncertainty-aware framework for individual treatment effect estimation with proxy-based latent confounder recovery.

The experiments evaluate IntervalGP-VAE under synthetic, semi-synthetic, ablation, robustness, scalability, and baseline-comparison settings. The main focus is to assess point-estimation accuracy, interval calibration, interval sharpness, latent confounder recovery, robustness to proxy degradation, and computational scalability.

---

## Overview

IntervalGP-VAE is designed for causal inference settings where the true latent confounder is unobserved but multiple proxy variables are available. The model learns a latent representation from proxies, estimates individual treatment effects, and propagates proxy-level uncertainty into calibrated ITE intervals.

The experimental notebooks are organized around the following research questions:

1. Which latent prior or GP regularizer gives the best accuracy-calibration trade-off?
2. How scalable is the IntervalGP-VAE framework when the training size increases?
3. Is the method sensitive to the choice of GP kernel?
4. Which uncertainty source contributes most to ITE interval calibration?
5. How does the proxy-count threshold (k \geq 2d + 4) behave empirically?
6. How does IntervalGP-VAE compare with other causal ITE baselines?
7. How robust is the method when proxy quality or proxy assumptions are degraded?
8. How does IntervalGP-VAE perform on the semi-synthetic IHDP benchmark?

---

## Notebook List

### 1. TMLR IntervalGP-VAE Latent Prior Ablation

This notebook compares different latent prior and regularization strategies while keeping the IntervalGP-VAE architecture and training framework fixed.

Compared variants include:

* **A0**: Standard VAE prior
* **A1**: Input-dependent Gaussian prior
* **A2**: Per-sample IntervalGP regularizer
* **A3-NLL**: Joint mini-batch GP negative log-likelihood regularizer
* **A3-KL**: Joint mini-batch GP KL regularizer
* **A4**: Sparse inducing-point GP variants with different inducing sizes

Main purpose:

* Identify which latent prior or GP regularizer provides the best balance between PEHE, ATE error, coverage, interval width, interval score, latent recovery, and runtime.

---

### 2. TMLR IntervalGP-VAE Scalability Analysis

This notebook evaluates the computational scalability of IntervalGP-VAE under different GP regularization strategies.

Compared variants include:

* Per-sample IntervalGP
* Sparse inducing IntervalGP with different inducing-point sizes

Main purpose:

* Study how runtime, memory, PEHE, coverage, and latent recovery change as the training size increases.
* Evaluate whether sparse inducing-point approximations can improve scalability while preserving treatment-effect accuracy.

---

### 3. TMLR IntervalGP-VAE Kernel Ablation

This notebook studies the sensitivity of IntervalGP-VAE to different GP kernel choices.

Compared kernels include:

* RBF
* Matérn-1/2
* Matérn-3/2
* Matérn-5/2
* Rational Quadratic
* Linear + RBF

Main purpose:

* Test whether IntervalGP-VAE depends strongly on a specific kernel.
* Evaluate PEHE, ATE error, coverage, width, NLL, interval score, latent recovery, and runtime across kernels.

---

### 4. TMLR IntervalGP-VAE ITE Interval Source Ablation

This notebook compares different sources of ITE interval uncertainty.

Compared variants include:

* **B0: Point only**

  * Uses only the point ITE estimate.
  * Sets lower bound equal to upper bound.

* **B1: Encoder MC latent interval**

  * Samples latent (U) from the encoder posterior.
  * Constructs ITE intervals from Monte Carlo quantiles.

* **B2: Latent-GP smoothed MC interval**

  * Applies GP smoothing to the latent representation.
  * Constructs ITE intervals from GP-smoothed latent samples.

* **B3: FullGP GP-ITE posterior**

  * Uses encoder MC samples to construct training ITE pseudo-observations.
  * Applies latent GP smoothing.
  * Fits full GP posteriors over lower, upper, and mean ITE functions.

Main purpose:

* Identify which uncertainty component contributes most to calibrated and useful ITE intervals.
* Test whether the final FullGP GP-ITE posterior improves the accuracy-calibration-width trade-off.

---

### 5. TMLR IntervalGP-VAE Proxy-Count Threshold Analysis

This notebook studies the relationship between proxy count (k) and latent dimension (d).

The experiment evaluates the sufficient proxy-count condition:

```text
k >= 2d + 4
```

The notebook includes both:

* Sub-threshold cases: (k < 2d + 4)
* At/above-threshold cases: (k \geq 2d + 4)

Main purpose:

* Test whether the theoretical proxy-count threshold behaves like a sharp finite-sample boundary or a conservative structural guideline.
* Evaluate how PEHE, ATE error, coverage, interval width, and latent recovery change as (k) and (d) vary.

---

### 6. TMLR IntervalGP-VAE Extended Baseline Comparison

This notebook compares IntervalGP-VAE against several causal ITE estimation baselines on the 24 synthetic proxy-based causal settings.

Compared methods include:

* IntervalGP-VAE
* Conformalized TEDVAE
* CEVAE-like baseline
* DeepPCL-like baseline
* BART/BCF-style tree ensemble
* Conformalized TARNet

Main purpose:

* Provide a broader empirical comparison against latent-variable, proxy-learning, representation-learning, tree-ensemble, and conformalized neural baselines.
* Evaluate point-estimation accuracy, interval calibration, interval width, interval score, and runtime.

---

### 7. TMLR IntervalGP-VAE Proxy Robustness Analysis

This notebook evaluates the robustness of IntervalGP-VAE under degraded proxy conditions.

The experiment includes three settings:

* **D1: Proxy noise degradation**

  * Gradually increases proxy noise.

* **D2: Proxy informativeness / redundancy degradation**

  * Reduces the number of informative proxies.
  * Allows non-informative proxies to become pure noise or redundant copies.

* **D3: Conditional independence violation**

  * Adds shared noise across proxies to violate conditional independence.

Main purpose:

* Test whether IntervalGP-VAE degrades gracefully when proxy quality becomes worse.
* Explain why semi-synthetic or real-world settings may show lower coverage when proxy assumptions are only partially satisfied.

---

### 8. TMLR IntervalGP-VAE Semi-Synthetic IHDP Reproduction

This notebook reproduces the semi-synthetic IHDP experiment and compares IntervalGP-VAE with conformalized TEDVAE over 100 IHDP replications.

Main purpose:

* Evaluate IntervalGP-VAE on a realistic semi-synthetic causal benchmark.
* Compare PEHE, ATE error, empirical ITE coverage, interval length, and runtime against TEDVAE.

Dataset path:

```text
./IHDP_b/
```

Compared methods:

* IntervalGP-VAE
* Conformalized TEDVAE
---

## Main Model Configuration

Most synthetic experiments use the following IntervalGP-VAE configuration:

```python
chosen_version = "u_aux"

INTERVALGPVAE_LATENT_DIM = 1
INTERVALGPVAE_HIDDEN_DIM = 64
CAUSAL_HEAD_HIDDEN_DIM = 64

GP_LENGTHSCALE = 7
GP_VARIANCE = 2.0
GP_NOISE = 1e-4

BATCH_SIZE = 128
JOINT_EPOCHS = 200
HEAD_EPOCHS = 100
VAE_REFINE_EPOCHS = 50

LR_JOINT = 1e-3
LR_HEAD = 1e-3
LR_VAE_REFINE = 1e-4
WEIGHT_DECAY = 1e-5

ITE_CI = 0.90
Z_VALUE_90 = 1.6448536269514722
```

The IHDP notebook uses a separate tuned configuration:

```python
latent_dim = 4
hidden_dim = 64
head_hidden_dim = 192

gp_lengthscale = 1.85
gp_variance = 3
gp_noise = 0.012

joint_epochs = 650
head_epochs = 250
vae_refine_epochs = 75
```

---

## Training Framework

Most notebooks use the same three-stage training framework.

### Stage 1: Joint Training

The VAE and causal head are trained jointly using proxy reconstruction loss, latent regularization, and outcome prediction loss.

### Stage 2: Causal Head Refinement

The VAE is frozen, and only the causal head is trained using the encoder posterior mean.

### Stage 3: VAE Refinement

The causal head is frozen, and the VAE is refined with causal loss to improve the latent representation for treatment-effect estimation.

---

## Inference Framework

The main IntervalGP-VAE inference pipeline uses:

1. Encoder posterior inference for latent (U)
2. Monte Carlo sampling of ITEs
3. Latent GP posterior smoothing over proxy space
4. ITE-space FullGP posterior prediction
5. Construction of 90% ITE intervals

---

## Evaluation Metrics

Across experiments, the following metrics are reported:

* **PEHE**

  * Precision in estimating heterogeneous treatment effects.

* **ATE error**

  * Absolute error in average treatment effect estimation.

* **Coverage**

  * Empirical coverage of the nominal 90% ITE interval.

* **Interval width**

  * Average length of the ITE interval.

* **Interval score**

  * Proper scoring rule combining sharpness and calibration.

* **Latent recovery error**

  * Error between recovered latent representation and true latent confounder after alignment.

* **Latent correlation**

  * Correlation between recovered latent representation and true latent confounder.

* **Runtime**

  * Training and/or total computational time.

---

## Dependencies

The notebooks require the following main Python packages:

```text
numpy
pandas
torch
matplotlib
scikit-learn
scipy
pyro-ppl
```

For TEDVAE experiments, the following are also required:

```text
pyro
tedvae_gpu.py
causal_datasets
IHDP_b/
```

The TEDVAE baseline is automatically skipped in some notebooks if Pyro or the TEDVAE implementation is unavailable.

---

## Dataset Preparation

### Synthetic Experiments

Synthetic experiments generate data internally. No external dataset is required.

### IHDP Experiment

For the IHDP reproduction notebook, place the IHDP data folder at:

```text
./IHDP_b/
```

The notebook loads IHDP using:

```python
from causal_datasets import IHDP
```

Only continuous IHDP covariates are used as proxy variables in the current implementation.

---

## Running the Notebooks

Each notebook can be run independently.

For full experiments, keep:

```python
MAX_ITERATIONS = None
```

For quick debugging, use a smaller value, for example:

```python
MAX_ITERATIONS = 2
```

For the IHDP notebook, the full run uses:

```python
SEARCH_REPS = list(range(1, 101))
```

For quick debugging, use:

```python
SEARCH_REPS = [1, 2]
```

## Notes on Reproducibility

The notebooks set random seeds for NumPy, PyTorch, Python random, and Pyro where applicable.

However, exact reproducibility may still depend on:

* CPU vs GPU execution
* PyTorch version
* Pyro version
* BLAS/LAPACK backend
* CUDA availability
* TEDVAE implementation details

For maximum reproducibility, run all experiments on the same hardware and software environment.

---

## Notes on Computational Cost

Some notebooks use FullGP posterior inference. FullGP computations can become expensive for large training sets because of cubic scaling in the number of GP reference points.

If computation becomes slow, set:

```python
MAX_GP_POINTS = 300
```

This uses a bounded GP reference set while keeping the posterior inference framework unchanged.

---

## Interpretation

These experiments should be interpreted together.

The synthetic experiments evaluate IntervalGP-VAE under controlled identifiable proxy regimes.

The ablation studies isolate the contributions of:

* latent prior regularization,
* GP kernel choice,
* interval uncertainty source,
* proxy count,
* proxy quality,
* proxy-assumption violations.

The IHDP experiment tests the method in a more realistic semi-synthetic setting where the proxy assumptions are only partially satisfied.

Therefore, lower empirical coverage on IHDP should be interpreted in light of partial proxy-assumption violation rather than as a failure of the interval-construction mechanism under the identifiable synthetic regime.

---

## Citation

If using these experiments, cite the corresponding IntervalGP-VAE manuscript:

```text
IntervalGP-VAE: Uncertainty-Aware Individual Treatment Effect Estimation
via Identifiable Proxy-Based Latent Confounder Recovery.
```
