# ForGAN — Probabilistic Financial Forecasting with Conditional GANs

Code for probabilistic future forecasting of financial processes using GAN architectures. Trains conditional generative adversarial networks on simulated trajectories from the **Black-Scholes** and **Heston** models, and evaluates the quality of the generated distributions against analytical ground truths probability density functions.

Three GAN variants are provided based on [Forgan](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8717640) and [Scoring Rule GAN](https://www.jmlr.org/papers/volume25/23-0038/23-0038.pdf): a standard **Conditional GAN** (CGAN), a **Wasserstein GAN with gradient penalty** (CWGAN-GP), and a discriminator-free **Scoring-Rule ForGAN** (SRForGAN). Each able to use either a plain MLP or an RNN (LSTM/GRU) as its generator backbone.

---

## Table of Contents

- [What the project does](#what-the-project-does)
- [Why it is useful](#why-it-is-useful)
- [Project structure](#project-structure)
- [ML models overview](#ml-models-overview)
  - [GAN architectures](#gan-architectures)
  - [Custom RNN layers](#custom-rnn-layers)
  - [Data simulators](#data-simulators)
- [Quick start](#quick-start)
- [Training pipeline](#training-pipeline)
- [Evaluation utilities](#evaluation-utilities)

---

## What the project does

Given a **look-back window** of observed log-prices `[X_{t-N}, …, X_t]`, the generator learns to sample from the conditional distribution `P(X_{t+1} | X_{t-N:t})`. At inference time, the model can draw arbitrarily many samples from this distribution, yielding a full probabilistic forecast rather than a single point estimate.

The training data are synthetic trajectories produced by closed-form stochastic differential equations (Black-Scholes GBM and the Heston stochastic-volatility model). Because the true conditional PDF is known analytically (or via fast numerical inversion), the quality of the generated distributions can be measured objectively with JS divergence, KS distance, Hellinger distance, and Wasserstein / Earth-Mover distance.

---

## Why it is useful

- **Model-agnostic probabilistic forecast**: any continuous target can be forecast as a full distribution, not just a point.
- **Three complementary training objectives**: adversarial (CGAN), Wasserstein + gradient penalty (CWGAN-GP), or scoring-rule minimisation without a discriminator (SRForGAN).
- **Flexible generator backbone**: swap a plain MLP for a multi-layer LSTM or GRU with optional LayerNorm, controlled by a single keyword argument.
- **Scalable to large datasets**: a custom binary format and `BinaryDataset` enable memory-efficient streaming of datasets that exceed RAM, with multi-worker DataLoader support.
- **HPC-ready training**: `train_large.py` includes SLURM detection, Automatic Mixed Precision (AMP), gradient checkpointing, epoch-level checkpoints, and automatic job-chaining for wall-time-limited clusters.
---

## Project structure

```
.
├── utilities.py              # DataSimulator (Black-Scholes), BinaryDataset, metrics
├── heston_data_simulator.py  # HestonSimulator — Heston SDE + frFFT PDF inversion
├── layers.py                 # Custom LayerNorm LSTM and GRU cells (single & multi-layer)
├── GANComponents.py          # Generator and Discriminator building blocks (MLP + RNN)
├── myCGAN.py                 # Conditional GAN trainer (BCE or Energy Score loss)
├── myCWGAN.py                # Wasserstein CGAN trainer (WGAN-GP)
├── mySRForGAN.py             # Scoring-Rule ForGAN (Energy Score, no discriminator)
├── train_large.py            # End-to-end HPC pipeline for million-path Heston datasets
└── test_fft.py               # frFFT validation: BS characteristic function → Gaussian PDF
```

---

## ML models overview

### GAN architectures

All three GAN variants share the same generator and discriminator building blocks from `GANComponents.py` and are found in the trainer classes below.

#### `MyCGAN` — Conditional GAN &nbsp;(`myCGAN.py`)

Standard conditional GAN where the generator receives a noise vector `z` concatenated with the condition window `c` and the discriminator scores `(x, c)` pairs. Supports:

- **BCE loss** (`torch.nn.BCEWithLogitsLoss`) — classic adversarial training.
- **Hybrid Energy Score + adversarial loss** — adds a prequential scoring-rule term to the generator loss so the generator also minimises CRPS.
- **`n_critic`** — number of discriminator updates per generator step.


#### `MyCWGAN` — Wasserstein Conditional GAN &nbsp;(`myCWGAN.py`)

Inherits `MyCGAN` and replaces the adversarial loss with the **Wasserstein distance + gradient penalty** (WGAN-GP). The critic outputs an unbounded real-valued score. Key differences from the parent class:

- `lambda_gp` (default 10) controls the gradient-penalty coefficient.
- `n_critic = 5` by default (more critic updates per generator step).
- Built-in **early stopping** on the Wasserstein distance.

#### `MySRForGAN` — Scoring-Rule ForGAN &nbsp;(`mySRForGAN.py`)

Implements the prequential **Energy Score** objective from [Pacchiardi et al. (2024)](https://www.jmlr.org/papers/volume25/23-0038/23-0038.pdf) *"Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization"*.

---

### Custom RNN layers

Found in `layers.py`. All standard PyTorch RNN modules normalise activations with BatchNorm, which performs poorly on sequential data. The custom cells here use **LayerNorm** instead, applied to the gate pre-activations before the nonlinearities.

| Class | Description |
|---|---|
| `LayerNormLSTM` | Single LSTM cell with LayerNorm on the concatenated gate vector. |
| `MultiLayerNormLSTM` | Stacked `LayerNormLSTM` cells with inter-layer dropout. |
| `LayerNormGRU` | Single GRU cell with separate LayerNorm on the reset/update gates and the candidate state. |
| `MultiLayerNormGRU` | Stacked `LayerNormGRU` cells with inter-layer dropout. |

These are consumed automatically by `RnnGenerator` and `RnnDiscriminator` in `GANComponents.py` when `use_layer_norm=True` is passed.

---

### Data simulators

#### `DataSimulator` — Black-Scholes GBM &nbsp;(`utilities.py`)

Simulates `J` independent log-price trajectories under Geometric Brownian Motion:

```
dX_t = (μ − ½σ²) dt + σ dW_t
```

Parameters `X₀`, `μ`, and `σ` are sampled per-trajectory from user-supplied ranges.

```python
from utilities import DataSimulator

sim = DataSimulator(X0_range=(0.0, 1.0), mu_range=(0.0, 0.1),
                    sigma_range=(0.1, 0.6), T=1.0, N=252,
                    n_simulations=1000, seed=42)
sim.get_paths()
pdf = sim.get_pdf(n_steps_ahead=1, n_bins=200)  # (J, 200)
```

`get_pdf` returns either a histogram of the analytical Gaussian CDF (`n_bins > 0`), raw `(mean, std)` parameters (`n_bins=None`) or Monte Carlo simulation (`n_bins = 0`). Data can be persisted to a compact binary format with `save_binary_file` / `load_binary_file`.

#### `HestonSimulator` — Heston Stochastic Volatility &nbsp;(`heston_data_simulator.py`)

Extends `DataSimulator` to the **Heston (1993)** model:

```
dX_t = (μ − ½v_t) dt + √v_t  dW_t^X
dv_t = κ(θ − v_t) dt + σ_v √v_t dW_t^v,
⟨dW^X, dW^v⟩ = ρ dt
```

```python
from heston_data_simulator import HestonSimulator

sim = HestonSimulator(
    X0_range=(0.0, 1.0), mu_range=(0.05, 0.05),
    v0_range=(0.04, 0.09), kappa_range=(1.0, 3.0),
    theta_range=(0.04, 0.09), sigma_v_range=(0.3, 0.6),
    rho_range=(-0.8, -0.3), T=1.0, N=252,
    n_simulations=500, seed=42, scheme='milstein'
)
sim.get_paths()
pdf = sim.get_pdf(n_steps_ahead=1, n_bins=200, Nfft=4096, eta=0.25)
```

A Monte-Carlo fallback (`mc_sims` argument) and a side-by-side visual validation (`plot_pdf_comparison`) are also provided.

---

## Quick start

```python
from utilities import DataSimulator, prepare_data
from mySRForGAN import MySRForGAN
from torch.utils.data import DataLoader

# 1. Simulate training data
sim = DataSimulator(X0_range=(0.0, 1.0), mu_range=(0.0, 0.1),
                    sigma_range=(0.1, 0.6), T=1.0, N=22,
                    n_simulations=50_000, seed=0)
sim.get_paths()

# targets: next log-price (J, 1)  |  conditions: path history (J, N)
targets    = sim.paths[:, -1:]
conditions = sim.paths[:, :-1]
dataset, _, _ = prepare_data(targets, conditions)

# 2. Build and train a SRForGAN with LSTM generator
model = MySRForGAN(max_epoch=50, n_samples_sr=10, use_amp=False)
model.set_generator(condition_size=22, output_dim=1,
                    hidden_dim_rnn=64, n_layers=2, rnn_layer='lstm')
model.train(dataset)

# 3. Generate probabilistic forecasts
import torch
c = torch.tensor(conditions[:8], dtype=torch.float32)
samples = model.generate(c, n_samples=500)   # (8, 500) forecast draws
```

---

## Training pipeline

`train_large.py` is the production entry point for datasets that exceed RAM (e.g. 1 million Heston paths). It:

1. **Generates** Heston paths in chunks and streams them to a flat binary file via `DataSimulator.save_binary_file`.
2. **Streams** the file at training time through `BinaryDataset` (O(1) random-access, DataLoader-safe).
3. Optionally runs an **Optuna hyperparameter search** (`RUN_TUNING=True`).
4. **Trains** `MySRForGAN` with AMP and SLURM-aware job chaining.
5. **Saves** generator weights, architecture config JSON, and training config JSON.

---

## Evaluation utilities

All metric helpers live in `utilities.py`.

| Function | Description |
|---|---|
| `get_error_metrics(true, generated)` | Returns JS distance, Hellinger distance, Total Variation, and Earth Mover distance between two sets of probability distributions. |
| `compare_simulated_pdfs(true, generated)` | Discretises continuous simulation draws onto a shared histogram grid for distribution comparison. |
| `compute_js(generated, true)` | Jensen-Shannon distance for a batch of distribution pairs. |
| `ks_test_gan_cdf(generated, true)` | KS statistic and p-value comparing generated vs. true CDFs. |
| `plot_bin_dist(trues, preds, bins)` | Side-by-side bar chart of true vs. generated histogram bins, with optional Normal overlay. |
| `analyze_error_distribution(csv)` | Violin-plot summary of per-bin errors saved to a CSV by a model evaluation run. |

The frFFT inversion accuracy can be independently verified by running `test_fft.py`, which feeds the analytical Black-Scholes characteristic function through `HestonSimulator._frfft_batch` and compares the recovered density against the exact Gaussian on the same quadrature grid.
