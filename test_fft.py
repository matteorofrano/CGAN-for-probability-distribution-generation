"""
test_frfft_bs.py
────────────────
Validates the frFFT inversion by feeding it the Black-Scholes characteristic
function and comparing the recovered PDF against the theoretical Gaussian
from DataSimulator.get_pdf().

Black-Scholes characteristic function
──────────────────────────────────────
Under GBM, the one-step-ahead log-price increment is Gaussian:

    X_{t+τ} | X_t  ~  N(μ_bs, σ²_bs)

where:
    μ_bs  = X_t + (μ - ½σ²)τ
    σ²_bs = σ² τ

The characteristic function is:

    φ_BS(u) = exp( iu·μ_bs  -  ½ σ²_bs u² )

Inversion (Gil-Pelaez / frFFT)
───────────────────────────────
    f(x_k) ≈ (η/π) Re[ Σ_{j=0}^{N-1}  w_j · φ_BS(jη) · e^{-i jη x_k} ]

where w_j are Simpson weights and x_k = b + k·λ,  λ = 2πα/η.

⚠ Aliasing note (α = 2/Nfft)
────────────────────────────
With α = 2/N, the frFFT computes y[k] = DFT[fft_in][2k], aliasing the
second half of the x-grid onto the first: raw integral ≈ 2.  This is
harmless — Heston's get_pdf() normalises the bins after histogram;
we do the same on the fine grid.

Comparison strategy
───────────────────
The frFFT grid spacing λ ≈ 0.012 is COARSER than what a tight histogram
would need for a narrow Gaussian (σ_bs ≈ 0.02–0.08). Re-binning with
N_BINS >> Nfft grid points in the support leaves most bins empty.

We therefore compare directly on the frFFT fine x-grid (both as continuous
densities normalised over the support), which is the most numerically honest
approach: same quadrature points, same spacing.

DataSimulator.get_pdf() is then validated separately: we call it with
n_bins=None (returns mean/std) and check those parameters match.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import jensenshannon

from utilities import DataSimulator
from heston_data_simulator import HestonSimulator   # for _frfft_batch


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Black-Scholes characteristic function
# ─────────────────────────────────────────────────────────────────────────────

def bs_cf(u: np.ndarray,
          X0: np.ndarray,
          mu: np.ndarray,
          sigma: np.ndarray,
          tau: float) -> np.ndarray:
    """
    φ_BS(u) = exp(iu·μ_bs - ½ σ²_bs u²)   shape (J, Nfft)
    """
    iu     = 1j * u[np.newaxis, :]
    mu_bs  = (X0 + (mu - 0.5 * sigma**2) * tau)[:, np.newaxis]
    var_bs = (sigma**2 * tau)[:, np.newaxis]
    return np.exp(iu * mu_bs - 0.5 * var_bs * (u**2)[np.newaxis, :])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Experiment setup
# ─────────────────────────────────────────────────────────────────────────────

SEED          = 42
J             = 8
N             = 50
T             = 1.0
N_STEPS_AHEAD = 1

Nfft        = 4096
eta         = 0.25
alpha_frfft = 2.0 / Nfft
lam         = 2.0 * np.pi * alpha_frfft / eta

X_WIDTH = 5.0     # half-width of support in std devs

tau = N_STEPS_AHEAD * (T / N)

print("=" * 65)
print("frFFT validation — Black-Scholes CF  →  Gaussian PDF")
print("=" * 65)
print(f"  J={J}, N={N}, T={T} yr,  τ={tau:.6f} yr")
print(f"  Nfft={Nfft},  η={eta},  α={alpha_frfft:.6f},  λ={lam:.6f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BS simulation + get_pdf() moment check
# ─────────────────────────────────────────────────────────────────────────────

sim = DataSimulator(
    X0_range    = (0.0, 1.0),
    mu_range    = (0.0, 0.1),
    sigma_range = (0.1, 0.6),
    T=T, N=N, n_simulations=J, seed=SEED,
)
sim.get_paths()

# Analytical parameters of the next-step Gaussian
bs_mean = sim.X_T + (sim.mu - 0.5 * sim.sigma**2) * tau
bs_std  = sim.sigma * np.sqrt(tau)

# DataSimulator.get_pdf with n_bins=None returns (mean, std) columns
pdf_ds = sim.get_pdf(n_steps_ahead=N_STEPS_AHEAD, n_bins=None)  # (J, 2)
get_pdf_mean = pdf_ds[:, 0]
get_pdf_std  = pdf_ds[:, 1]

print("Step A — DataSimulator.get_pdf() moment validation:")
print(f"  {'k':>3}  {'bs_mean':>12}  {'get_pdf mean':>14}  {'|Δ|':>10}"
      f"  {'bs_std':>12}  {'get_pdf std':>13}  {'|Δ|':>10}")
print("-" * 90)
for k in range(J):
    print(f"  {k:3d}  {bs_mean[k]:12.7f}  {get_pdf_mean[k]:14.7f}  "
          f"{abs(bs_mean[k]-get_pdf_mean[k]):10.2e}"
          f"  {bs_std[k]:12.7f}  {get_pdf_std[k]:13.7f}  "
          f"{abs(bs_std[k]-get_pdf_std[k]):10.2e}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  frFFT inversion with the BS characteristic function
# ─────────────────────────────────────────────────────────────────────────────

j_arr = np.arange(Nfft, dtype=np.float64)
u     = j_arr * eta

# Simpson weights
w         = np.ones(Nfft, dtype=np.float64)
w[1:-1:2] = 4.0
w[2:-2:2] = 2.0
w        /= 3.0

phi    = bs_cf(u, sim.X_T, sim.mu, sim.sigma, tau)           # (J, Nfft)
b      = bs_mean - X_WIDTH * bs_std                           # (J,)
phase  = np.exp(-1j * u[np.newaxis, :] * b[:, np.newaxis])   # (J, Nfft)
fft_in = w[np.newaxis, :] * phi * phase                      # (J, Nfft)

Y        = HestonSimulator._frfft_batch(fft_in, alpha_frfft)
pdf_fine = np.maximum((eta / np.pi) * np.real(Y), 0.0)       # (J, Nfft)
x_grid   = b[:, np.newaxis] + j_arr[np.newaxis, :] * lam     # (J, Nfft)

# Aliasing diagnostic
raw_mass = pdf_fine.sum(axis=1) * lam
print(f"Raw integral (α=2/N → aliasing doubles mass, expect ≈ 2):")
print("  " + "  ".join(f"t{k}:{raw_mass[k]:.4f}" for k in range(J)))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Restrict to support and normalise  (≡ Heston's histogram+normalise step)
# ─────────────────────────────────────────────────────────────────────────────

x_lo = (bs_mean - X_WIDTH * bs_std)[:, np.newaxis]
x_hi = (bs_mean + X_WIDTH * bs_std)[:, np.newaxis]
mask = (x_grid >= x_lo) & (x_grid <= x_hi)           # (J, Nfft)

# frFFT density — zero outside support, then normalise to unit mass
pdf_frfft = pdf_fine.copy()
pdf_frfft[~mask] = 0.0
norm_f = pdf_frfft.sum(axis=1, keepdims=True) * lam
norm_f = np.where(norm_f == 0, 1.0, norm_f)
pdf_frfft /= norm_f          # (J, Nfft)  units: 1/x

# Analytical Gaussian density at the SAME x-grid points
pdf_gauss = norm.pdf(x_grid,
                     loc=bs_mean[:, np.newaxis],
                     scale=bs_std[:, np.newaxis])    # (J, Nfft)
pdf_gauss[~mask] = 0.0
norm_g = pdf_gauss.sum(axis=1, keepdims=True) * lam
norm_g = np.where(norm_g == 0, 1.0, norm_g)
pdf_gauss /= norm_g          # (J, Nfft)  units: 1/x


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Moments from the fine grid
# ─────────────────────────────────────────────────────────────────────────────

frfft_mean = (pdf_frfft * x_grid).sum(axis=1) * lam
frfft_var  = (pdf_frfft * (x_grid - frfft_mean[:, None])**2).sum(axis=1) * lam
frfft_std  = np.sqrt(np.maximum(frfft_var, 0.0))

print("Step B — frFFT fine-grid moment accuracy:")
print(f"  {'k':>3}  {'frfft_mean':>12}  {'bs_mean':>12}  {'|Δmean|':>10}"
      f"  {'frfft_std':>12}  {'bs_std':>12}  {'|Δstd|':>10}")
print("-" * 85)
for k in range(J):
    print(f"  {k:3d}  {frfft_mean[k]:12.7f}  {bs_mean[k]:12.7f}  "
          f"{abs(frfft_mean[k]-bs_mean[k]):10.2e}"
          f"  {frfft_std[k]:12.7f}  {bs_std[k]:12.7f}  "
          f"{abs(frfft_std[k]-bs_std[k]):10.2e}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Distribution metrics — directly on the fine x-grid
# ─────────────────────────────────────────────────────────────────────────────
#
# Convert density (1/x) to probability masses (dimensionless) by × lam,
# then normalise.  Both PDFs share the exact same quadrature points.

def _safe_js(p, q):
    sp, sq = p.sum(), q.sum()
    if sp == 0 or sq == 0:
        return np.nan
    return float(jensenshannon(p / sp, q / sq, base=2.0))

def _ks(p, q):
    """KS statistic on discrete probability masses."""
    return float(np.max(np.abs(np.cumsum(p / p.sum())
                               - np.cumsum(q / q.sum()))))

def _tv(p, q):
    sp, sq = p.sum(), q.sum()
    return float(0.5 * np.sum(np.abs(p / sp - q / sq)))

def _hellinger(p, q):
    sp, sq = p.sum(), q.sum()
    pn, qn = p / sp, q / sq
    return float(np.sqrt(0.5 * np.sum((np.sqrt(pn) - np.sqrt(qn))**2)))


metrics = {m: [] for m in ["ks", "js", "tv", "hellinger"]}

for k in range(J):
    in_sup = mask[k]
    pf = pdf_frfft[k, in_sup] * lam     # probability mass at each grid point
    pg = pdf_gauss[k, in_sup] * lam

    metrics["ks"].append(_ks(pf, pg))
    metrics["js"].append(_safe_js(pf, pg))
    metrics["tv"].append(_tv(pf, pg))
    metrics["hellinger"].append(_hellinger(pf, pg))


print("Step C — Distribution metrics on frFFT fine grid (support ±5σ)")
print(f"  (comparing {int(mask.sum(1).mean()):.0f} active grid points per trajectory "
      f"at spacing λ={lam:.5f})")
print()
print(f"  {'k':>3}  {'KS':>9}  {'JS dist':>9}  {'TV':>9}  {'Hellinger':>11}")
print("-" * 50)
for k in range(J):
    print(f"  {k:3d}  {metrics['ks'][k]:9.6f}  {metrics['js'][k]:9.6f}"
          f"  {metrics['tv'][k]:9.6f}  {metrics['hellinger'][k]:11.6f}")
print("-" * 50)
for name, vals in metrics.items():
    print(f"  mean {name:12s}: {np.nanmean(vals):.6f}")
print()
print("Thresholds:  KS/TV/Hellinger < 0.02 → excellent | < 0.05 → good")
print("             JS dist          < 0.05 → excellent | < 0.10 → good")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plots
# ─────────────────────────────────────────────────────────────────────────────

ncols = min(J, 4)
nrows = int(np.ceil(J / ncols))

# ── Figure 1: fine-grid density overlay ──────────────────────────────────────
fig1, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
axes = np.array(axes).flatten()

for k in range(J):
    ax = axes[k]
    in_sup = mask[k]
    xk  = x_grid[k, in_sup]
    fk  = pdf_frfft[k, in_sup]
    gk  = pdf_gauss[k, in_sup]

    ax.plot(xk, gk, color="steelblue", lw=2.5,
            label=f"Gaussian N({bs_mean[k]:.4f}, {bs_std[k]:.5f}²)")
    ax.plot(xk, fk, color="crimson", lw=1.5, ls="--",
            label="frFFT (BS CF)")

    ax.set_title(
        f"Traj {k}  μ={sim.mu[k]:.3f}, σ={sim.sigma[k]:.3f}\n"
        f"KS={metrics['ks'][k]:.5f}   JS={metrics['js'][k]:.5f}   "
        f"TV={metrics['tv'][k]:.5f}",
        fontsize=8,
    )
    ax.set_xlabel(r"$X_{t+\tau}$ (log-price)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)

for extra in axes[J:]:
    fig1.delaxes(extra)

fig1.suptitle(
    f"frFFT validation — BS CF  "
    f"(τ={tau:.5f} yr,  Nfft={Nfft},  η={eta},  α={alpha_frfft:.5f})\n"
    f"Densities compared directly on the frFFT fine x-grid",
    fontsize=10, y=1.02,
)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/frfft_bs_density_overlay.png",
            dpi=150, bbox_inches="tight")
print("Saved → frfft_bs_density_overlay.png")


# ── Figure 2: residual (frFFT - Gaussian) ────────────────────────────────────
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows))
axes2 = np.array(axes2).flatten()

for k in range(J):
    ax = axes2[k]
    in_sup = mask[k]
    xk     = x_grid[k, in_sup]
    resid  = pdf_frfft[k, in_sup] - pdf_gauss[k, in_sup]

    ax.plot(xk, resid, color="purple", lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Traj {k}  residual = frFFT − Gaussian\n"
                 f"max|residual|={np.max(np.abs(resid)):.2e}", fontsize=8)
    ax.set_xlabel(r"$X_{t+\tau}$", fontsize=8)
    ax.set_ylabel("f_frfft − f_gauss", fontsize=8)

for extra in axes2[J:]:
    fig2.delaxes(extra)

fig2.suptitle(
    "Residuals: frFFT density − analytical Gaussian density (support only)",
    fontsize=10, y=1.02,
)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/frfft_bs_residuals.png",
            dpi=150, bbox_inches="tight")
print("Saved → frfft_bs_residuals.png")

plt.show()