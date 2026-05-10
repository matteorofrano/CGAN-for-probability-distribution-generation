"""
mySRForGAN.py
=============
Scoring-Rule-based Conditional Generative Model for probabilistic forecasting.

Implements the prequential Energy Score objective from:
  Pacchiardi et al. (2024) - "Probabilistic Forecasting with Generative Networks
  via Scoring Rule Minimization", JMLR 25.

Key idea (Section 3.1)
----------------------
Instead of the adversarial min-max game, the generator is trained to minimise
the *prequential scoring rule*:

    φ̂ = argmin_φ  Σ_{t=k}^{T-l}  S( P_φ(·|y_{t-k+1:t}),  y_{t+l} )

For the Energy Score (Appendix B.2.1, β=1):

    S_E(P, y) = 2·E[‖X − y‖] − E[‖X − X'‖],   X, X' ~ P

Unbiased estimate with m generator draws (Appendix C.1.1):

    Ŝ_E({x_j}, y) = (2/m)·Σ_j ‖x_j − y‖  −  (1/(m(m-1)))·Σ_{j≠k} ‖x_j − x_k‖

Because P_φ is parametrised via z ~ N(0,I) → x = h_φ(z; c), the reparametrisation
trick makes the gradient of this estimate w.r.t. φ unbiased (no critic required).

Advantages over CGAN / CWGAN
------------------------------
1. No discriminator → zero adversarial instability / mode collapse.
2. Training loss is directly meaningful → clean early stopping on a validation set.
3. Fewer hyper-parameters: no λ_gp, no n_critic, no critic architecture.
4. Consistent under mixing/stationarity (Theorem 3 of the paper).

Usage
-----
    from mySRForGAN import MySRForGAN

    model = MySRForGAN(scoring_rule='energy', n_samples_sr=10)
    model.set_generator(condition_size=22, output_dim=1)
    model.train(train_dataset, val_data=val_dataset)
"""

import copy
import time
import json
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Re-use generator definitions and the parent class scaffolding
from myCGAN import MyCGAN
from utilities import TensorDataset, pd


# ---------------------------------------------------------------------------
# Scoring rule functions (all differentiable through autograd)
# ---------------------------------------------------------------------------

def energy_score(samples: torch.Tensor, y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Unbiased Energy Score estimate (paper eq. C.1 / B.2.1).

    Parameters
    ----------
    samples : (batch, m, output_dim)  — m draws from P_φ(·|c)
    y       : (batch, output_dim)     — observed next step
    beta    : float, exponent in (0,2).  β=1 recovers CRPS for scalars.

    Returns
    -------
    score : (batch,)   — per-sample score (lower = better)
    """
    batch_size, m, d = samples.shape
    y_exp = y.unsqueeze(1)                               # (batch, 1, d)

    # ---- Term 1 : 2·E_X[ ‖X − y‖^β ]  ----
    diff_xy = torch.norm(samples - y_exp, dim=-1)        # (batch, m)
    if beta != 1.0:
        diff_xy = diff_xy ** beta
    term1 = 2.0 * diff_xy.mean(dim=1)                   # (batch,)

    # ---- Term 2 : E_{X,X'}[ ‖X − X'‖^β ] (unbiased U-statistic) ----
    xi = samples.unsqueeze(2)                            # (batch, m, 1, d)
    xj = samples.unsqueeze(1)                            # (batch, 1, m, d)
    diff_xx = torch.norm(xi - xj, dim=-1)               # (batch, m, m)
    if beta != 1.0:
        diff_xx = diff_xx ** beta

    # Mask diagonal to exclude j==k pairs
    eye = torch.eye(m, device=samples.device).unsqueeze(0)   # (1, m, m)
    diff_xx = diff_xx * (1.0 - eye)
    term2 = diff_xx.sum(dim=(1, 2)) / (m * (m - 1))    # (batch,)

    return term1 - term2                                 # (batch,)


def kernel_score(
    samples: torch.Tensor, y: torch.Tensor, bandwidth: float = 1.0
) -> torch.Tensor:
    """
    Unbiased Kernel Score estimate with a Gaussian kernel (paper eq. B.2.2 / C.1.2).

    k(x, x') = exp( −‖x − x'‖² / (2γ²) )

    Parameters
    ----------
    samples   : (batch, m, output_dim)
    y         : (batch, output_dim)
    bandwidth : γ (scalar; tune via median heuristic on the validation set)

    Returns
    -------
    score : (batch,)
    """
    batch_size, m, d = samples.shape
    gamma2 = 2.0 * bandwidth ** 2

    # ---- Term 1 : E[k(X, X')] (unbiased) ----
    xi = samples.unsqueeze(2)                                   # (batch, m, 1, d)
    xj = samples.unsqueeze(1)                                   # (batch, 1, m, d)
    dist2_xx = ((xi - xj) ** 2).sum(dim=-1)                    # (batch, m, m)
    k_xx = torch.exp(-dist2_xx / gamma2)

    eye = torch.eye(m, device=samples.device).unsqueeze(0)
    k_xx = k_xx * (1.0 - eye)
    term1 = k_xx.sum(dim=(1, 2)) / (m * (m - 1))              # (batch,)

    # ---- Term 2 : −2·E[k(X, y)] ----
    y_exp = y.unsqueeze(1)                                      # (batch, 1, d)
    dist2_xy = ((samples - y_exp) ** 2).sum(dim=-1)            # (batch, m)
    k_xy = torch.exp(-dist2_xy / gamma2)
    term2 = -2.0 * k_xy.mean(dim=1)                           # (batch,)

    return term1 + term2                                        # (batch,)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MySRForGAN(MyCGAN):
    """
    Scoring-Rule-based Conditional ForGAN.

    Trains the generator to minimise the prequential Energy Score (or Kernel
    Score, or their sum) instead of an adversarial objective.  No discriminator
    is used or needed.

    Parameters
    ----------
    max_epoch          : int    — maximum training epochs
    batch_size         : int    — mini-batch size
    z_noise_dim        : int    — latent noise dimension (must match condition window)
    n_samples_sr       : int    — m in the paper: generator draws per condition.
                                  Paper uses 10 during training; as few as 3 works
                                  (Appendix F.3.2).
    scoring_rule       : str    — 'energy' | 'kernel' | 'energy_kernel'
    beta               : float  — exponent for Energy Score (default 1; must be in (0,2))
    kernel_bandwidth   : float  — γ for Gaussian kernel (tune via median heuristic)
    lr_g               : float  — generator learning rate
    early_stopping_patience   : int   — epochs without val improvement before stopping
    early_stopping_min_delta  : float — minimum improvement threshold
    name               : str    — model name
    """

    def __init__(
        self,
        max_epoch: int = 200,
        batch_size: int = 256,
        z_noise_dim: int = 252,
        n_samples_sr: int = 10,
        scoring_rule: str = "energy",
        beta: float = 1.0,
        kernel_bandwidth: float = 1.0,
        lr_g: float = 1e-3,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-5,
        name: str = "SRForGAN",
    ):
        # Initialise parent but disable the discriminator path
        super().__init__(
            max_epoch=max_epoch,
            batch_size=batch_size,
            n_critic=1,
            z_noise_dim=z_noise_dim,
            loss_fn=None,     # no adversarial loss
            lr_g=lr_g,
            lr_d=lr_g,        # unused but parent expects it
            name=name,
        )
        if scoring_rule not in ("energy", "kernel", "energy_kernel"):
            raise ValueError(
                f"scoring_rule must be 'energy', 'kernel' or 'energy_kernel'. "
                f"Got '{scoring_rule}'."
            )
        if not (0.0 < beta < 2.0):
            raise ValueError(f"beta must be in (0, 2). Got {beta}.")

        self.n_samples_sr = n_samples_sr
        self.scoring_rule = scoring_rule
        self.beta = beta
        self.kernel_bandwidth = kernel_bandwidth
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

    # ------------------------------------------------------------------
    # Core SR computation
    # ------------------------------------------------------------------

    def _generate_samples(self, c: torch.Tensor) -> torch.Tensor:
        """
        Draw m samples from P_φ(·|c) via the reparametrisation trick.

        Parameters
        ----------
        c : (batch_size, condition_size)

        Returns
        -------
        samples : (batch_size, m, output_dim)
        """
        batch_size = c.size(0)
        m = self.n_samples_sr

        # Expand condition: (batch*m, condition_dim)
        c_expanded = c.unsqueeze(1).expand(-1, m, -1).reshape(batch_size * m, -1)

        # Independent noise draws for each sample (reparametrisation trick)
        z = torch.randn(batch_size * m, self.z_dim, device=c.device)

        raw = self.G(c_expanded, z)                          # (batch*m, output_dim)
        output_dim = raw.size(-1)
        return raw.view(batch_size, m, output_dim)           # (batch, m, d)

    def _compute_sr_loss(self, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean scoring rule over the batch.

        Parameters
        ----------
        c : (batch, condition_dim)  — observation window  y_{t-k+1:t}
        y : (batch, output_dim)     — next-step realisation  y_{t+l}

        Returns
        -------
        loss : scalar tensor (differentiable w.r.t. generator params)
        """
        samples = self._generate_samples(c)                  # (batch, m, d)

        if self.scoring_rule == "energy":
            sr = energy_score(samples, y, beta=self.beta)
        elif self.scoring_rule == "kernel":
            sr = kernel_score(samples, y, bandwidth=self.kernel_bandwidth)
        else:  # energy_kernel  (Lemma 4: sum of proper SRs is strictly proper)
            sr = energy_score(samples, y, beta=self.beta) + \
                 kernel_score(samples, y, bandwidth=self.kernel_bandwidth)

        return sr.mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(                                              # type: ignore[override]
        self,
        data: TensorDataset,
        val_data: Optional[TensorDataset] = None,
        save_history: bool = False,
    ):
        """
        Train the generator via scoring rule minimisation.

        The discriminator (self.D) is **not used**.  Only the generator (self.G)
        is optimised with plain Adam — no adversarial loop, no gradient penalty.

        Parameters
        ----------
        data         : TensorDataset of (y, c) pairs — training set
        val_data     : optional validation TensorDataset for early stopping
                       (strongly recommended: the SR loss is a meaningful metric,
                       unlike the GAN generator loss).
        save_history : bool — if True, saves per-epoch losses to CSV

        Notes
        -----
        Data format expected: each item is (target, condition).
          target    y_{t+l}        shape (output_dim,)
          condition y_{t-k+1:t}    shape (condition_size,)
        This matches the existing TensorDataset layout used by MyCGAN.
        """
        if self.G is None:
            raise RuntimeError(
                "Generator not defined. Call set_generator() first."
            )
        if not isinstance(data, torch.utils.data.Dataset):
            raise TypeError(
                f"data must be a TensorDataset. Got {type(data)}."
            )

        self.G.to(self.DEVICE)

        train_loader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = (
            DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
            if val_data is not None
            else None
        )

        # Plain Adam — no betas tuning needed for SR training
        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.lr_g)

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None

        history = []
        start = time.time()

        for epoch in range(self.max_epoch):
            # ---- Training pass ----
            self.G.train()
            train_loss_sum = 0.0

            for y_batch, c_batch in train_loader:
                y_batch = y_batch.to(self.DEVICE).float()
                c_batch = c_batch.to(self.DEVICE).float()

                G_opt.zero_grad()
                loss = self._compute_sr_loss(c_batch, y_batch)
                loss.backward()
                G_opt.step()

                train_loss_sum += loss.item()

            avg_train = train_loss_sum / len(train_loader)

            # ---- Validation pass ----
            avg_val: Optional[float] = None
            if val_loader is not None:
                self.G.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for y_v, c_v in val_loader:
                        y_v = y_v.to(self.DEVICE).float()
                        c_v = c_v.to(self.DEVICE).float()
                        val_loss_sum += self._compute_sr_loss(c_v, y_v).item()
                avg_val = val_loss_sum / len(val_loader)

            # ---- Logging ----
            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                val_str = f"  Val SR: {avg_val:.6f}" if avg_val is not None else ""
                print(
                    f"Epoch {epoch:4d}/{self.max_epoch}  "
                    f"Train SR: {avg_train:.6f}{val_str}"
                )

            history.append(
                {"epoch": epoch, "train_sr": avg_train, "val_sr": avg_val}
            )

            # ---- Early stopping (only if validation data provided) ----
            if avg_val is not None:
                if avg_val < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = avg_val
                    patience_counter = 0
                    best_state = copy.deepcopy(self.G.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(
                            f"\nEarly stopping at epoch {epoch}. "
                            f"Best val SR: {best_val_loss:.6f}"
                        )
                        if best_state is not None:
                            self.G.load_state_dict(best_state)
                        break

        elapsed = time.time() - start
        print(f"\nTraining finished in {elapsed:.1f}s.")

        if save_history and history:
            pd.DataFrame(history).to_csv(
                f"{self.MODEL_NAME}_sr_history.csv", index=False
            )
            print(f"History saved to {self.MODEL_NAME}_sr_history.csv")

        return history

    # ------------------------------------------------------------------
    # Save / load  (generator only — no discriminator)
    # ------------------------------------------------------------------

    def save_models(self, save_dir: str = "./models"):
        """
        Save the generator and a config file.
        The discriminator is not saved because it does not exist in SR training.
        """
        import os, json
        os.makedirs(save_dir, exist_ok=True)

        gen_path = os.path.join(save_dir, f"{self.MODEL_NAME}_generator.pth")
        self.save_generator(gen_path)

        cfg = {
            "max_epoch":      self.max_epoch,
            "batch_size":     self.batch_size,
            "z_dim":          self.z_dim,
            "n_samples_sr":   self.n_samples_sr,
            "scoring_rule":   self.scoring_rule,
            "beta":           self.beta,
            "kernel_bandwidth": self.kernel_bandwidth,
            "model_name":     self.MODEL_NAME,
            "lr_g":           self.lr_g,
        }
        cfg_path = os.path.join(save_dir, f"{self.MODEL_NAME}_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"SR config saved to {cfg_path}")

    def load_models(self, load_dir: str = "./models"):
        """
        Load the generator (and config) from *load_dir*.
        No discriminator is loaded.
        """
        import os, json
        from GANComponents import MyGenerator

        cfg_path = os.path.join(load_dir, f"{self.MODEL_NAME}_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.z_dim            = cfg.get("z_dim",            self.z_dim)
            self.n_samples_sr     = cfg.get("n_samples_sr",     self.n_samples_sr)
            self.scoring_rule     = cfg.get("scoring_rule",     self.scoring_rule)
            self.beta             = cfg.get("beta",             self.beta)
            self.kernel_bandwidth = cfg.get("kernel_bandwidth", self.kernel_bandwidth)
            print(f"SR config loaded from {cfg_path}")

        gen_path = os.path.join(load_dir, f"{self.MODEL_NAME}_generator.pth")
        self.G = MyGenerator.load(gen_path, device=self.DEVICE)
        self.z_dim = self.G.latent_size

    # ------------------------------------------------------------------
    # Kernel bandwidth helper (median heuristic, Appendix E.1)
    # ------------------------------------------------------------------

    def tune_kernel_bandwidth(self, data: TensorDataset, n_pairs: int = 2000) -> float:
        """
        Set kernel bandwidth γ to the median of pairwise observation distances
        (median heuristic, Appendix E.1 of the paper).

        Call this *before* training when using scoring_rule='kernel' or
        'energy_kernel'.

        Parameters
        ----------
        data    : TensorDataset — the training dataset
        n_pairs : int           — number of observation pairs to sample

        Returns
        -------
        γ : float — the bandwidth (also stored as self.kernel_bandwidth)
        """
        loader = DataLoader(data, batch_size=n_pairs, shuffle=True)
        y_batch, _ = next(iter(loader))
        y_np = y_batch.numpy().reshape(len(y_batch), -1)

        # Sample pairwise distances
        idx = np.random.choice(len(y_np), size=(min(n_pairs, len(y_np)), 2), replace=True)
        dists = np.linalg.norm(y_np[idx[:, 0]] - y_np[idx[:, 1]], axis=-1)
        gamma = float(np.median(dists))

        print(f"Median heuristic → kernel_bandwidth γ = {gamma:.6f}")
        self.kernel_bandwidth = gamma
        return gamma

    # ------------------------------------------------------------------
    # Disable the discriminator interface (not needed)
    # ------------------------------------------------------------------

    def set_discriminator(self, *args, **kwargs):
        """Not used in SR training — override to no-op with a warning."""
        import warnings
        warnings.warn(
            "MySRForGAN does not use a discriminator. "
            "set_discriminator() has no effect.",
            UserWarning,
        )

    def set_critic(self, *args, **kwargs):
        """Alias for set_discriminator — no-op."""
        self.set_discriminator(*args, **kwargs)


# ---------------------------------------------------------------------------
# Quick usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from utilities import DataSimulator, prepare_data

    # --- Simulate GBM paths ---
    sim = DataSimulator(
        X0_range=(1.0, 1.0),
        mu_range=(0.0, 0.0),
        sigma_range=(0.1, 0.3),
        T=1.0,
        N=252,
        n_simulations=50_000,
        seed=42,
    )
    paths = sim.get_paths()           # log-returns trajectory (condition window)
    targets = sim.get_pdf(n_steps_ahead=1)   # next log-return (scalar target)

    train_data, _, _ = prepare_data(targets, paths)

    # --- Build and train model ---
    model = MySRForGAN(
        max_epoch=200,
        batch_size=512,
        z_noise_dim=paths.shape[1],
        n_samples_sr=10,
        scoring_rule="energy",   # start with Energy Score; try 'energy_kernel' next
        lr_g=1e-3,
        early_stopping_patience=20,
    )
    model.set_generator(
        condition_size=paths.shape[1],
        output_dim=1,                   # scalar next log-return
        hidden_dims=[128, 256, 128],
    )

    history = model.train(train_data, save_history=True)