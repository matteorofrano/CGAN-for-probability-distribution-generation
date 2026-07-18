"""
heston_simulator.py
───────────────────
Extends the Black-Scholes DataSimulator to the Heston (1993) stochastic
volatility model.

Model (log-price form)
──────────────────────
    dX_t = (μ - ½v_t) dt + √v_t  dW_t^X
    dv_t = κ(θ - v_t) dt + σ_v √v_t dW_t^v
    ⟨dW_t^X, dW_t^v⟩ = ρ dt

PDF computation
───────────────
The conditional PDF of X_{T+τ} | (X_T, v_T) is obtained by numerically
inverting the Heston characteristic function using the **Fractional FFT**
(Chourdakis 2004, based on Bluestein 1970).

The key advantage of the frFFT over the standard FFT is that the frequency
step η and the x-domain spacing λ are decoupled:

    α = η λ / (2π)   ← fractional parameter (freely chosen)

This lets you choose a fine price grid (small λ) without being forced to
use a large η, and vice-versa.

Characteristic function
───────────────────────
The "Little Trap" formulation (Albrecher et al. 2007) is used to avoid the
branch-cut discontinuity present in the original Heston (1993) formula:

    d      = √[(κ - ρ σ_v i u)² + σ_v²(u² + i u)]    (Re(d) ≥ 0)
    g̃      = (κ - ρ σ_v i u - d) / (κ - ρ σ_v i u + d)
    A(u,τ) = (κθ/σ_v²)[(κ - ρ σ_v i u - d)τ - 2 ln((1 - g̃ e^{-dτ})/(1 - g̃))]
    B(u,τ) = (v_0/σ_v²)(κ - ρ σ_v i u - d)(1 - e^{-dτ})/(1 - g̃ e^{-dτ})
    φ(u)   = exp(i u X_0 + i u μ τ + A + B)

Discretisation scheme
─────────────────────
The variance SDE uses the *Milstein scheme with full truncation*:

    v̂_{n+1} = max(v_n, 0)           ← full truncation before evaluation
    v_{n+1}  = v_n + κ(θ - v̂_n)Δt + σ_v √v̂_n ΔW_v + ¼σ_v²(ΔW_v² - Δt)
    v_{n+1}  = max(v_{n+1}, 0)      ← full truncation after step

The log-price uses Euler-Maruyama:
    X_{n+1} = X_n + (μ - ½v̂_n)Δt + √v̂_n ΔW_X

"""

from __future__ import annotations

from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from utilities import DataSimulator

from cir_obj   import cir_obj
from cir_evol  import QT_cir_evol
from heston_evol import mc_heston


# ─────────────────────────────────────────────────────────────────────────────
class HestonSimulator(DataSimulator):
    """
    Simulate log-price trajectories under the Heston stochastic-volatility
    model and compute the conditional next-step PDF via the fractional FFT.

    Parameters
    ----------
    X0_range : tuple (lo, hi) or list of floats
        Initial log-price.  Tuple → uniform sampling; list → fixed values.
    mu_range : tuple or list
        Risk-neutral drift μ (same convention as DataSimulator).
    v0_range : tuple or list
        Initial instantaneous variance v_0.
    kappa_range : tuple or list
        Mean-reversion speed κ > 0.
    theta_range : tuple or list
        Long-run variance θ > 0.
    sigma_v_range : tuple or list
        Vol-of-vol σ_v > 0.  Feller condition: 2κθ > σ_v² ensures v_t > 0 a.s.
    rho_range : tuple or list
        Spot-vol correlation ρ ∈ (-1, 1); typically negative for equity.
    T : float
        Time horizon in years.
    N : int
        Number of time steps.
    n_simulations : int
        Number J of independent trajectories to simulate.
    seed : int or None
        RNG seed for reproducibility.
    scheme : {'milstein', 'euler'}
        Discretisation scheme for the variance SDE.

    Attributes set after get_paths()
    ---------------------------------
    paths          : np.ndarray (J, N+1)   log-price trajectories
    variance_paths : np.ndarray (J, N+1)   variance trajectories
    X_T            : np.ndarray (J,)       terminal log-prices
    v_T            : np.ndarray (J,)       terminal variances

    Attributes set after get_pdf()
    --------------------------------
    pdf            : np.ndarray (J, n_bins) or (J, 2)
    bins           : np.ndarray (n_bins+1,)
    means          : np.ndarray (J,)       first moment (computed from frFFT grid)
    stds           : np.ndarray (J,)       second moment
    """

    def __init__(
        self,
        X0_range: Union[Tuple[float, float], List[float]],
        mu_range: Union[Tuple[float, float], List[float]],
        v0_range: Union[Tuple[float, float], List[float]],
        kappa_range: Union[Tuple[float, float], List[float]],
        theta_range: Union[Tuple[float, float], List[float]],
        sigma_v_range: Union[Tuple[float, float], List[float]],
        rho_range: Union[Tuple[float, float], List[float]],
        T: float,
        N: int,
        n_simulations: int,
        seed: int | None = None,
        scheme: str = "milstein",):
        
        super().__init__(
            X0_range=X0_range,
            mu_range=mu_range,
            sigma_range=sigma_v_range,   # unused; kept for parent compatibility
            T=T,
            N=N,
            n_simulations=n_simulations,
            seed=seed,
        )

        if scheme not in ("milstein", "euler"):
            raise ValueError("scheme must be 'milstein' or 'euler'.")
        self.scheme = scheme

        # Heston-specific parameter ranges
        self.v0_range = v0_range
        self.kappa_range = kappa_range
        self.theta_range = theta_range
        self.sigma_v_range = sigma_v_range
        self.rho_range = rho_range

        # Sampled Heston parameters — filled by get_paths()
        self.v0: np.ndarray | None = None
        self.kappa: np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.sigma_v: np.ndarray | None = None
        self.rho: np.ndarray | None = None

        # Additional output arrays
        self.variance_paths: np.ndarray | None = None
        self.v_T: np.ndarray | None = None

    # ── Parameter sampling ────────────────────────────────────────────────────

    def _sample(self, range_: Union[Tuple, List]) -> np.ndarray:
        """
        Sample n_simulations values from range_.

        If range_ is a tuple (lo, hi) → draw uniformly from [lo, hi].
        If range_ is a list            → use as-is (fixed values).
        """
        if isinstance(range_, tuple):
            lo, hi = float(range_[0]), float(range_[1])
            return self.rng.uniform(lo, hi, size=self.n_simulations)
        elif isinstance(range_, list):
            arr = np.array(range_, dtype=float)
            if arr.ndim == 0 or len(arr) != self.n_simulations:
                # broadcast scalar / short list to all simulations
                return np.full(self.n_simulations, arr.flat[0])
            return arr
        else:
            raise TypeError(
                f"Parameter range must be a tuple or list, got {type(range_)}."
            )

    # ── Simulation ────────────────────────────────────────────────────────────

    def get_paths(self, get_proxy_n: int = 0) -> np.ndarray:
        """
        Simulate J log-price and variance trajectories under the Heston model.

        The variance uses *full-truncation Milstein* (default) or
        *full-truncation Euler-Maruyama*.  The log-price uses Euler-Maruyama.

        Parameters
        ----------
        get_proxy_n : int
            If > 0, return [√v_T, X_{T-get_proxy_n+1}, …, X_T] instead of
            the full path matrix (same convention as DataSimulator).

        Returns
        -------
        paths : np.ndarray (J, N+1)   log-price paths (or proxy array if
                get_proxy_n > 0).
        """
        J  = self.n_simulations
        N  = self.N
        dt = self.dt
        sqrt_dt = np.sqrt(dt)

        # ── Sample per-trajectory parameters ─────────────────────────────────
        self.X0      = self._sample(self.X0_range)
        self.mu      = self._sample(self.mu_range)
        self.v0      = self._sample(self.v0_range)
        self.kappa   = self._sample(self.kappa_range)
        self.theta   = self._sample(self.theta_range)
        self.sigma_v = self._sample(self.sigma_v_range)
        self.rho     = self._sample(self.rho_range)

        # with the parent's save/load binary methods
        self.sigma = self.v0

        # ── Allocate storage ──────────────────────────────────────────────────
        X = np.zeros((J, N + 1), dtype=np.float64)
        V = np.zeros((J, N + 1), dtype=np.float64)
        X[:, 0] = self.X0
        V[:, 0] = self.v0

        # ── Time-stepping ─────────────────────────────────────────────────────
        for n in range(N):
            # Correlated Brownian increments
            Z1 = self.rng.standard_normal(size=J)
            Z2 = self.rng.standard_normal(size=J)
            dW_X = sqrt_dt * Z1
            dW_v = sqrt_dt * (self.rho * Z1 + np.sqrt(1.0 - self.rho**2) * Z2)

            # Full truncation: use max(V_n, 0) inside diffusion coefficients
            v_n     = np.maximum(V[:, n], 0.0)
            sqrt_vn = np.sqrt(v_n)

            # ── Log-price: Euler-Maruyama ─────────────────────────────────────
            X[:, n + 1] = (
                X[:, n]
                + (self.mu - 0.5 * v_n) * dt
                + sqrt_vn * dW_X
            )

            # ── Variance: Milstein or Euler with full truncation ──────────────
            drift_v = self.kappa * (self.theta - v_n) * dt
            diff_v  = self.sigma_v * sqrt_vn * dW_v

            if self.scheme == "milstein":
                # Milstein correction: ¼ σ_v² (ΔW_v² - Δt)
                milstein = 0.25 * self.sigma_v**2 * (dW_v**2 - dt)
                V[:, n + 1] = V[:, n] + drift_v + diff_v + milstein
            else:
                V[:, n + 1] = V[:, n] + drift_v + diff_v

            # Apply full truncation after the step
            V[:, n + 1] = np.maximum(V[:, n + 1], 0.0)

        # ── Store results ─────────────────────────────────────────────────────
        self.paths          = X
        self.variance_paths = V
        self.X_T            = X[:, -1]
        self.v_T            = V[:, -1]

        if get_proxy_n > 0:
            # Return [√v_T | last get_proxy_n log-price values]
            return np.column_stack([
                np.sqrt(self.v_T),
                self.paths[:, -get_proxy_n:],
            ])
        return self.paths

    # ── Heston characteristic function ────────────────────────────────────────

    @staticmethod
    def _heston_cf(
        u: np.ndarray,
        tau: float,
        X0: np.ndarray,
        v0: np.ndarray,
        mu: np.ndarray,
        kappa: np.ndarray,
        theta: np.ndarray,
        sigma_v: np.ndarray,
        rho: np.ndarray,
    ) -> np.ndarray:
        """
        Heston characteristic function — "Little Trap" formulation.

        φ(u; τ) = exp(iu X_0 + iu μ τ + A(u,τ) + B(u,τ) v_0)

        where:
            d      = √[(κ - ρ σ_v iu)² + σ_v²(u² + iu)]   with Re(d) ≥ 0
            g̃      = (κ - ρ σ_v iu - d) / (κ - ρ σ_v iu + d)
            A(u,τ) = (κθ/σ_v²)[(κ - ρ σ_v iu - d)τ
                              - 2 ln((1 - g̃ e^{-dτ}) / (1 - g̃))]
            B(u,τ) = (v_0/σ_v²)(κ - ρ σ_v iu - d)
                              × (1 - e^{-dτ}) / (1 - g̃ e^{-dτ})

        Parameters
        ----------
        u        : real 1-D array (Nfft,)  frequency grid
        tau      : float                   forecast horizon (in years)
        X0, v0, mu, kappa, theta, sigma_v, rho : 1-D arrays (J,)
                   per-trajectory parameters

        Returns
        -------
        phi : complex array (J, Nfft)
        """
        # Promote u to (1, Nfft) and parameters to (J, 1) for broadcasting
        iu  = 1j * u[np.newaxis, :]                    # (1, Nfft)
        X0  = X0     [:, np.newaxis].astype(complex)   # (J, 1)
        v0  = v0     [:, np.newaxis].astype(complex)
        mu  = mu     [:, np.newaxis].astype(complex)
        kap = kappa  [:, np.newaxis].astype(complex)
        th  = theta  [:, np.newaxis].astype(complex)
        sv  = sigma_v[:, np.newaxis].astype(complex)
        rh  = rho    [:, np.newaxis].astype(complex)

        # ξ = κ - ρ σ_v iu
        xi = kap - rh * sv * iu                        # (J, Nfft)

        # d = √[ξ² + σ_v²(u² + iu)]  — enforce Re(d) ≥ 0
        u2 = (u**2 + u * 1j)[np.newaxis, :]           # (1, Nfft), u² + iu
        d  = np.sqrt(xi**2 + sv**2 * u2)               # (J, Nfft)
        d  = np.where(np.real(d) >= 0.0, d, -d)        # stable branch

        # g̃ = (ξ - d) / (ξ + d)
        g_tilde = (xi - d) / (xi + d)                  # (J, Nfft)

        exp_dtau = np.exp(-d * tau)                     # e^{-dτ}
        denom    = 1.0 - g_tilde * exp_dtau             # 1 - g̃ e^{-dτ}

        # A term — uses Little Trap log argument (1 - g̃ e^{-dτ})/(1 - g̃)
        log_arg = denom / (1.0 - g_tilde)
        A = (kap * th / sv**2) * ((xi - d) * tau - 2.0 * np.log(log_arg))

        # B term
        B = (v0 / sv**2) * (xi - d) * (1.0 - exp_dtau) / denom

        phi = np.exp(iu * X0 + iu * mu * tau + A + B)  # (J, Nfft)
        return phi

    # ── Fractional FFT ────────────────────────────────────────────────────────

    @staticmethod
    def _frfft_batch(X: np.ndarray, alpha: float) -> np.ndarray:
        """
        Batched Fractional FFT via Bluestein's identity.

        Computes, for each row x of X:

            y[k] = Σ_{j=0}^{N-1}  x[j] · exp(-i 2π α j k),   k = 0,…,N-1

        This is the standard DFT when α = 1/N; for α ≠ 1/N the summation is
        over an **arbitrary** exponential grid — the "fractional" DFT.

        Algorithm (Bluestein 1970)
        --------------------------
        Using jk = [j² + k² - (j-k)²]/2:

            y[k] = e^{-iπα k²} · Σ_j [ x[j] e^{-iπα j²} ] · e^{iπα(j-k)²}

        The inner sum is a **linear convolution**, computed in O(M log M) via
        FFT on a zero-padded array of length M = next_power_of_2(2N-1).

        Parameters
        ----------
        X     : complex ndarray (batch, N)
        alpha : float   fractional parameter  α = η λ / (2π)

        Returns
        -------
        Y : complex ndarray (batch, N)
        """
        batch, N = X.shape
        # Minimum FFT length to avoid circular aliasing: M ≥ 2N−1
        M = int(2 ** np.ceil(np.log2(2 * N - 1)))

        j     = np.arange(N, dtype=np.float64)
        chirp = np.exp(1j * np.pi * alpha * j**2)   # e^{iπα j²}, shape (N,)

        # ── Phase-modulated input: a[j] = x[j] · e^{−iπα j²} ────────────────
        a = X * np.conj(chirp)[np.newaxis, :]        # (batch, N)

        # ── Zero-pad to length M ──────────────────────────────────────────────
        a_pad           = np.zeros((batch, M), dtype=complex)
        a_pad[:, :N]    = a

        # ── Convolution kernel h: h[j] = e^{iπα j²} (symmetric: h[-j]=h[j]) ─
        # Laid out for circular convolution:
        #   h_pad[0 : N]       = h[0], h[1], …, h[N-1]
        #   h_pad[M-N+1 : M]   = h[N-1], …, h[1]   (= h[-(N-1)], …, h[-1])
        # Zeros fill positions N … M-N (the "guard" zone).
        h_pad           = np.zeros(M, dtype=complex)
        h_pad[:N]       = chirp
        h_pad[M-N+1:]   = chirp[1:][::-1]           # h[-1], …, h[-(N-1)]

        # ── Circular convolution via FFT ──────────────────────────────────────
        A_fft = np.fft.fft(a_pad, axis=-1)           # (batch, M)
        H_fft = np.fft.fft(h_pad)                    # (M,)
        conv  = np.fft.ifft(A_fft * H_fft[np.newaxis, :], axis=-1)  # (batch, M)

        # ── Multiply by e^{−iπα k²} and extract first N outputs ──────────────
        Y = np.conj(chirp)[np.newaxis, :] * conv[:, :N]   # (batch, N)
        return Y

    # ── PDF via frFFT ─────────────────────────────────────────────────────────

    def get_pdf( # type: ignore
        self,
        n_steps_ahead: int,
        n_bins: int | None = None,
        P: np.ndarray | None = None,
        v: np.ndarray | None = None,
        mc_sims: int = 0,
        verbose: bool = False,
        Nfft: int = 4096,
        eta: float = 0.25,
        alpha_frfft: float | None = None,
        x_width: float = 5.0,
    ) -> np.ndarray:
        """
        Compute the conditional PDF of the log-price n_steps_ahead*dt ahead.

        The PDF is evaluated by numerically inverting the Heston characteristic
        function on a fine x-grid using the **Fractional FFT**.

        Inversion formula (Gil-Pelaez 1951)
        ------------------------------------
            f(x) = (η/π) Re[ Σ_{j=0}^{Nfft-1} w_j φ(j η) e^{-i j η x} ]

        where w_j are Simpson quadrature weights and η is the frequency step.
        The frFFT evaluates this simultaneously at all x_k = b + k λ,
        k = 0,…,Nfft-1, for any chosen spacing λ = 2π α / η.

        Parameters
        ----------
        n_steps_ahead : int
            Forecast horizon in units of dt.
        n_bins : int or None
            Number of histogram bins.  None → return raw moments (mean, std).
        P : np.ndarray (J,) or None
            Override starting log-prices; defaults to self.X_T.
        mc_sims : int
            If ≥ 2, use Monte-Carlo instead of frFFT (for validation).
        verbose : bool
        Nfft : int
            FFT length (power of 2 recommended).  Controls frequency resolution.
        eta : float
            Frequency-domain step η.  Larger η → narrower x domain.
        alpha_frfft : float or None
            Fractional parameter α = η λ / (2π).  Defaults to 2/Nfft,
            giving λ ≈ 0.03 for η=0.25, Nfft=4096 — far finer than the
            standard FFT spacing of 2π/(Nfft·η) ≈ 0.006.
        x_width : float
            Half-width of the per-trajectory x-grid in approximate std devs.

        Returns
        -------
        self.pdf : np.ndarray (J, n_bins) if n_bins is not None
                   np.ndarray (J, 2)      otherwise  (columns: mean, std)
        """
        if self.X_T is None or self.v_T is None:
            raise RuntimeError("Call get_paths() before get_pdf().")
        if self.kappa is None:
            raise RuntimeError("Heston parameters are not initialised.")

        J   = self.n_simulations
        tau = n_steps_ahead * self.dt

        # ── Starting conditions ───────────────────────────────────────────────
        X0 = (P if P is not None else self.X_T).copy()
        v0 = (v if v is not None else self.v_T).copy()

        # ── Monte-Carlo fallback ──────────────────────────────────────────────
        if mc_sims >= 2:
            return self._mc_pdf(X0=X0, v0=v0, n_bins=n_bins, mc_sims=mc_sims, n_steps=n_steps_ahead)

        # ── frFFT parameters ──────────────────────────────────────────────────
        if alpha_frfft is None:
            alpha_frfft = 2.0 / Nfft          # default: finer grid than std FFT

        lam = 2.0 * np.pi * alpha_frfft / eta  # x-domain spacing λ

        if verbose:
            print(f"[frFFT] Nfft={Nfft}, η={eta}, α={alpha_frfft:.6f}, "
                  f"λ={lam:.6f}, τ={tau:.4f}")

        # ── Per-trajectory grid lower bounds ─────────────────────────────────
        # Approximate centre and spread using long-run parameters
        approx_mean = X0 + (self.mu - 0.5 * self.theta) * tau       # (J,)
        approx_std  = np.sqrt(np.maximum(self.theta, 0.0) * tau)     # (J,)
        b = approx_mean - x_width * approx_std                        # (J,)

        # ── Frequency grid u_j = j η ──────────────────────────────────────────
        j_arr = np.arange(Nfft, dtype=np.float64)
        u     = j_arr * eta                    # (Nfft,)  real

        # ── Simpson quadrature weights ────────────────────────────────────────
        w       = np.ones(Nfft, dtype=np.float64)
        w[1:-1:2] = 4.0   # odd indices
        w[2:-2:2] = 2.0   # even indices (interior)
        w       /= 3.0

        # ── Heston characteristic function (J, Nfft) ──────────────────────────
        phi = self._heston_cf(
            u, tau, X0, v0,
            self.mu, self.kappa, self.theta, self.sigma_v, self.rho,
        )

        # ── Phase shift for the lower bound ───────────────────────────────────
        # e^{−i u_j b_k} for each trajectory k
        phase = np.exp(-1j * u[np.newaxis, :] * b[:, np.newaxis])   # (J, Nfft)

        # ── frFFT input: w_j · φ_j(u) · e^{−i u_j b} ────────────────────────
        fft_in = w[np.newaxis, :] * phi * phase                      # (J, Nfft)

        # ── Apply batched frFFT ────────────────────────────────────────────────
        # y[k] = Σ_j fft_in[j] · exp(-i 2π α j k)
        # f(x_k) ≈ (η/π) Re[y[k]]
        Y       = self._frfft_batch(fft_in, alpha_frfft)             # (J, Nfft)
        pdf_fine = (eta / np.pi) * np.real(Y)                        # (J, Nfft)
        pdf_fine = np.maximum(pdf_fine, 0.0)                         # clip artefacts

        # ── x-grid (J, Nfft) ──────────────────────────────────────────────────
        k_arr  = np.arange(Nfft, dtype=np.float64)
        x_grid = b[:, np.newaxis] + k_arr[np.newaxis, :] * lam       # (J, Nfft)

        self.pdf_fine   = pdf_fine   # (J, Nfft) raw density (not normalised)
        self.x_grid     = x_grid     # (J, Nfft) corresponding x-values
        self._frfft_lam = lam        # grid spacing (needed for normalisation)

        # ── Return raw moments if no binning requested ────────────────────────
        if n_bins is None or n_bins == 0:
            norm_const  = pdf_fine.sum(axis=1) * lam + 1e-30
            means = (pdf_fine * x_grid).sum(axis=1) * lam / norm_const
            vars_ = (pdf_fine * (x_grid - means[:, None])**2).sum(axis=1) * lam / norm_const
            self.means = means
            self.stds  = np.sqrt(np.maximum(vars_, 0.0))
            self.pdf   = np.column_stack((means, self.stds)).astype(np.float32)
            return self.pdf

        # ── Build shared global bin edges ─────────────────────────────────────
        if self.bins is not None:
            common_bins = self.bins
            n_bins      = len(common_bins) - 1
        else:
            global_lo   = (approx_mean - x_width * approx_std).min()
            global_hi   = (approx_mean + x_width * approx_std).max()
            common_bins = np.linspace(global_lo, global_hi, n_bins + 1)
            self.bins   = common_bins

        # ── Resample fine PDF grid onto common bins (rectangle rule) ──────────
        # np.histogram with pdf_fine * lam as weights gives probability mass
        probabilities = np.zeros((J, n_bins), dtype=np.float32)
        for j_idx in range(J):
            weights = pdf_fine[j_idx] * lam          # probability mass per cell
            hist, _ = np.histogram(
                x_grid[j_idx], bins=common_bins, weights=weights
            )
            probabilities[j_idx] = hist.astype(np.float32)

        # ── Normalise ─────────────────────────────────────────────────────────
        probabilities[probabilities < 1e-7] = 0.0
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        probabilities /= row_sums

        self.pdf = probabilities
        return self.pdf
    
    def get_smooth_curves(self,
                          bins_list: list,
                          n_plot: int = 500) -> list:
        """
        Return a list of (x_smooth, y_smooth) pairs — one per trajectory —
        
        Must be called AFTER ``get_pdf()`` (which stores ``self.pdf_fine``,
        ``self.x_grid`` and ``self._frfft_lam``).
 
        y_smooth is in **probability mass** units (density × bin_width)
 
        Parameters
        ----------
        bins_list : list of np.ndarray
            Per-trajectory bin edges returned as the third element of
            ``compare_simulated_pdfs()``.  Each entry may have a different
            width, so bin_width is recomputed inside the loop.
        n_plot : int
            Number of interpolation points for the smooth curve (default 500).
        """
        from scipy.interpolate import interp1d
 
        if self.pdf_fine is None or self.x_grid is None:
            raise RuntimeError("Call get_pdf() before get_smooth_curves().")
 
        lam    = self._frfft_lam
        curves = []
 
        for k in range(len(bins_list)):
            bin_width = float(bins_list[k][1] - bins_list[k][0])  # per-trajectory
            xk = self.x_grid[k]
            
            # Restrict to meaningful support
            x_lo = bins_list[k][0]
            x_hi = bins_list[k][-1]
            mask      = (xk >= x_lo) & (xk <= x_hi)
            if mask.sum() < 4:                          # too few pts → skip
                curves.append((xk, fk * bin_width))
                continue
 
            xk_s = xk[mask]
            fk_s  = self.pdf_fine[k][mask]
            norm_c = fk_s.sum() * lam + 1e-30
            fk_s  = fk_s / norm_c  
            x_plot = np.linspace(xk_s[0], xk_s[-1], n_plot)
            y_plot = np.maximum(
                interp1d(xk_s, fk_s, kind="cubic")(x_plot), 0.0
            )
            # Convert density → probability mass per bin
            curves.append((x_plot, y_plot * bin_width))
 
        return curves
    
    # ── Monte-Carlo PDF (reference / fallback) ────────────────────────────────

    def _mc_pdf(
        self,
        X0: np.ndarray,
        v0: np.ndarray,
        n_bins: int | None,
        mc_sims: int,
        n_steps: int,
        get_raw_terminal_values: bool = False
    ) -> np.ndarray:
        """
        Monte-Carlo estimate of the conditional Heston PDF.

        For each of the J trajectories mc_sims sample paths are evolved over
        tau = n_steps * dt years using:
          • QT_cir_evol  — Andersen (2007) Quadratic-Exponential scheme for the
                           CIR variance, which also returns the integrated
                           variance ∫v dt.  Never produces negative variance.
          • mc_heston    — exact log-price update given the variance path
                           (see heston_evol.py).

        The loop is now over J parameter sets (not mc_sims), so mc_sims paths
        per set are vectorised inside QT_cir_evol / mc_heston.
        """
        J       = self.n_simulations
        dt      = self.dt
        tau = n_steps*dt
        tf = np.array([0.0, tau])  

        terminal = np.zeros((J, mc_sims), dtype=np.float64)

        for j in range(J):
            cir_j = cir_obj(
                kappa = float(self.kappa[j]),
                sigma = float(self.sigma_v[j]),
                theta = float(self.theta[j]),
                ro    = float(v0[j]),
            )
            # vol shape (2, mc_sims): vol[0]=v0, vol[1]=v_tau
            # Ivol shape (2, mc_sims): Ivol[1] = ∫_0^tau v dt
            vol, Ivol = QT_cir_evol(self.rng, cir_j, tf, dt, mc_sims)

            # S shape (2, mc_sims), S[0]=1 (normalised); uses exact formula
            S = mc_heston(self.rng, vol, Ivol, cir_j, float(self.rho[j]), tf)

            # log-price = X0[j] + log(S_tau) + mu[j]*tau  (mu shift is additive)
            terminal[j] = X0[j] + np.log(np.maximum(S[-1], 1e-300)) + self.mu[j] * tau

        if get_raw_terminal_values:
            return terminal
        
        # ── Return raw moments if no binning ──────────────────────────────────
        if n_bins == 0:
            self.means = terminal.mean(axis=1)
            self.stds  = terminal.std(axis=1)
            self.pdf   = np.column_stack((self.means, self.stds)).astype(np.float32)
            return self.pdf

        # ── Histogram onto shared bins ────────────────────────────────────────
        probabilities = np.zeros((J, len(self.bins) - 1), dtype=np.float32)
        if n_bins is not None:
            hist_bins_input = self.bins if self.bins is not None else np.ceil(np.sqrt(mc_sims))
        else:
            hist_bins_input = np.ceil(np.sqrt(mc_sims))

        for j_idx in range(J):
            hist, bin_edges = np.histogram(terminal[j_idx], bins= hist_bins_input)
            total   = hist.sum()
            self.bin_edges = bin_edges
            if total > 0:
                probabilities[j_idx] = (hist / total).astype(np.float32)

        probabilities[probabilities < 1e-7] = 0.0
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        probabilities /= row_sums

        self.pdf = probabilities
        return self.pdf

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self):
        """
        Plot log-price paths, variance paths, and the terminal log-price
        histogram, mirroring the DataSimulator.plot() interface.
        """
        if self.paths is None or self.variance_paths is None:
            raise RuntimeError("Call get_paths() before plot().")

        time_grid = np.linspace(0, self.T, self.N + 1)
        plt.style.use("seaborn-v0_8-whitegrid")

        # ── Log-price paths ───────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

        axes[0].plot(time_grid, self.paths.T, lw=0.7, alpha=0.6)
        axes[0].set_title(
            f"{self.n_simulations} Heston Log-Price Trajectories", fontsize=14
        )
        axes[0].set_ylabel(r"Log price  $X_t = \ln S_t$", fontsize=11)

        # ── Variance paths ────────────────────────────────────────────────────
        axes[1].plot(time_grid, self.variance_paths.T, lw=0.7, alpha=0.6)
        axes[1].set_title(
            f"{self.n_simulations} Heston Variance Trajectories", fontsize=14
        )
        axes[1].set_xlabel("Time (years)", fontsize=11)
        axes[1].set_ylabel(r"Variance  $v_t$", fontsize=11)

        plt.tight_layout()
        plt.show()

        # ── Terminal log-price histogram ──────────────────────────────────────
        plt.figure(figsize=(8, 4))
        plt.hist(self.paths[:, -1], bins=min(100, self.n_simulations // 5 + 10))
        plt.title(r"Terminal log-price distribution  $X_T = \ln S_T$  (Heston)")
        plt.xlabel(r"$X_T$")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_pdf_comparison(
        self,
        trajectory_indices: list[int] | None = None,
        mc_sims: int = 20_000,
        n_steps_ahead: int = 1,
        Nfft: int = 4096,
        eta: float = 0.25,
        x_width: float = 12.0
    ):
        """
        Overlay the frFFT PDF against a Monte-Carlo histogram for a handful
        of trajectories — a quick visual validation of the inversion.

        Parameters
        ----------
        trajectory_indices : list of int or None
            Which trajectories to plot.  Defaults to the first 4.
        mc_sims : int
            Number of MC paths per trajectory for the reference histogram.
        n_steps_ahead : int
            Forecast horizon.
        Nfft, eta : frFFT parameters.
        """
        if self.X_T is None:
            raise RuntimeError("Call get_paths() first.")

        indices = trajectory_indices or list(range(min(4, self.n_simulations)))
        tau     = n_steps_ahead * self.dt
        dt      = self.dt
        sqrt_dt = np.sqrt(dt)

        ncols = min(len(indices), 2)
        nrows = int(np.ceil(len(indices) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        alpha_frfft = 2.0 / Nfft
        lam         = 2.0 * np.pi * alpha_frfft / eta

        u_arr = np.arange(Nfft, dtype=np.float64) * eta
        w     = np.ones(Nfft)
        w[1:-1:2] = 4.0
        w[2:-2:2] = 2.0
        w        /= 3.0

        for plot_i, j_idx in enumerate(indices):
            ax = axes[plot_i]

            # ── Monte-Carlo reference ─────────────────────────────────────────
            X_mc = np.full(mc_sims, self.X_T[j_idx])
            v_mc = np.full(mc_sims, self.v_T[j_idx])
            for _ in range(n_steps_ahead):
                Z1   = self.rng.standard_normal(mc_sims)
                Z2   = self.rng.standard_normal(mc_sims)
                dW_X = sqrt_dt * Z1
                dW_v = sqrt_dt * (
                    self.rho[j_idx] * Z1
                    + np.sqrt(1.0 - self.rho[j_idx]**2) * Z2
                )
                # Full truncation: clamp variance before using in coefficients
                v_pos = np.maximum(v_mc, 0.0)                   # BUG 3 FIX
                sv    = np.sqrt(v_pos)
                X_mc += (self.mu[j_idx] - 0.5 * v_pos) * dt + sv * dW_X
                v_mc += (
                    self.kappa[j_idx] * (self.theta[j_idx] - v_pos) * dt
                    + self.sigma_v[j_idx] * sv * dW_v
                    + 0.25 * self.sigma_v[j_idx]**2 * (dW_v**2 - dt)  # BUG 4 FIX
                )
                v_mc = np.maximum(v_mc, 0.0)

            ax.hist(
                X_mc, bins=80, density=True,
                alpha=0.5, color="steelblue", label="MC histogram"
            )

            # ── frFFT PDF ─────────────────────────────────────────────────────
            X0_j  = np.array([self.X_T[j_idx]])
            v0_j  = np.array([self.v_T[j_idx]])
            mu_j  = np.array([self.mu[j_idx]])
            kap_j = np.array([self.kappa[j_idx]])
            th_j  = np.array([self.theta[j_idx]])
            sv_j  = np.array([self.sigma_v[j_idx]])
            rh_j  = np.array([self.rho[j_idx]])

            phi = self._heston_cf(
                u_arr, tau, X0_j, v0_j, mu_j, kap_j, th_j, sv_j, rh_j
            )

            approx_std = np.sqrt(float(v0_j) * tau)
            b_j        = (float(X0_j)
                          + (float(mu_j) - 0.5 * float(v0_j)) * tau
                          - x_width * approx_std)

            phase  = np.exp(-1j * u_arr * b_j)
            fft_in = (w * phi[0] * phase)[np.newaxis, :]

            Y_j   = self._frfft_batch(fft_in, alpha_frfft)[0]
            pdf_j = np.maximum((eta / np.pi) * np.real(Y_j), 0.0)
            x_j   = b_j + np.arange(Nfft) * lam

            ax.plot(x_j, pdf_j, color="crimson", lw=1.8, label="frFFT PDF")
            ax.set_xlim(X_mc.min() - 0.05, X_mc.max() + 0.05)
            ax.set_title(
                f"Traj {j_idx}  "
                f"(κ={self.kappa[j_idx]:.2f}, v0={self.v_T[j_idx]:.3f}, sv={self.sigma_v[j_idx]:.2f},  θ={self.theta[j_idx]:.3f}, "
                f"ρ={self.rho[j_idx]:.2f})",
                fontsize=9,
            )
            ax.set_xlabel(r'$X_{T+\tau}$')
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        for extra in axes[len(indices):]:
            fig.delaxes(extra)

        plt.suptitle(
            f"Heston PDF Validation  (τ={tau:.4f} yr,  Nfft={Nfft},  η={eta})",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("HestonSimulator — quick demo")
    print("=" * 60)

    # ── Typical SPX-ish parameters ────────────────────────────────────────────
    sim = HestonSimulator(
        X0_range      = (0.0, 1.0),       # log-price start
        mu_range      = (0.05, 0.05),     # risk-free rate (constant)
        v0_range      = (0.04, 0.09),     # initial variance (vol ≈ 20–30%)
        kappa_range   = (1.0, 3.0),       # mean-reversion speed
        theta_range   = (0.04, 0.09),     # long-run variance
        sigma_v_range = (0.3, 0.6),       # vol-of-vol
        rho_range     = (-0.8, -0.3),     # spot-vol correlation (negative)
        T             = 1.0,
        N             = 252,              # daily steps
        n_simulations = 500,
        seed          = 42,
        scheme        = "milstein",
    )

    # ── Simulate paths ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sim.get_paths()
    print(f"get_paths()  → paths shape: {sim.paths.shape}  "       # type: ignore
          f"({time.perf_counter()-t0:.3f} s)")

    # ── Compute PDF via frFFT (binned) ────────────────────────────────────────
    t0 = time.perf_counter()
    pdf = sim.get_pdf(n_steps_ahead=10, n_bins=200,
                      Nfft=4096, eta=0.25, verbose=True)
    print(f"get_pdf()    → pdf shape:   {pdf.shape}  "
          f"({time.perf_counter()-t0:.3f} s)")

    # ── Raw moments (no binning) ──────────────────────────────────────────────
    sim.bins = None    # reset bins so get_pdf recomputes
    pdf_raw = sim.get_pdf(n_steps_ahead=10, n_bins=None, Nfft=4096)
    print(f"Raw moments  → shape: {pdf_raw.shape}  "
          f"(cols: mean, std)")
    print(f"  mean range : [{pdf_raw[:, 0].min():.4f}, {pdf_raw[:, 0].max():.4f}]")
    print(f"  std  range : [{pdf_raw[:, 1].min():.4f}, {pdf_raw[:, 1].max():.4f}]")

    # ── Save / load binary (inherited) ────────────────────────────────────────
    sim.bins = None
    sim.get_pdf(n_steps_ahead=10, n_bins=64)
    sim.save_binary_file("heston_demo")
    paths2, pdf2 = sim.load_binary_file("heston_demo")
    print(f"Binary round-trip OK: paths {paths2.shape}, pdf {pdf2.shape}")

    # ── Visual validation: frFFT PDF vs. MC histogram ─────────────────────────
    sim.plot_pdf_comparison(
        trajectory_indices=[0, 1, 2, 3],
        mc_sims=30_000,
        n_steps_ahead=10,
    )

    # ── Path plot (log-price + variance) ─────────────────────────────────────
    sim.plot()