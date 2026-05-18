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

Checkpoint system
-----------------
``train()`` now supports three new parameters for HPC job chaining:

    checkpoint_dir         : str | None
        Directory where .pt checkpoint files are written.
        Each file is named  checkpoint_epoch_{NNNN}.pt
        and contains: epoch index (= *next* epoch to run), model weights,
        optimizer state, GradScaler state, best_val_loss, patience_counter,
        and the full loss history so far.

    checkpoint_every       : int  (default 1)
        Save a checkpoint after every N completed epochs.

    resume_from            : str | None
        Path to a .pt checkpoint produced by a previous run.
        All training state is restored so the new job continues seamlessly.

    max_runtime_seconds    : float | None
        If set, the training loop monitors elapsed wall-clock time.
        When  elapsed + time_save_buffer_seconds >= max_runtime_seconds
        the loop saves a checkpoint and exits cleanly so the *next* SLURM
        job can resume from it.

    time_save_buffer_seconds : float  (default 300 = 5 minutes)
        Safety margin reserved for the final checkpoint write before the
        job's wall-time limit is hit.

GPU / HPC changes vs. original
-------------------------------
1. **AMP (Automatic Mixed Precision)** — new ``use_amp`` parameter (default True
   when CUDA is available).  Training forward+loss runs under
   ``torch.autocast("cuda", dtype=torch.float16)``; gradients are scaled with
   ``torch.cuda.amp.GradScaler`` to prevent underflow.  Typical throughput gain:
   1.5–2× on A100 / V100 / RTX series.  Set ``use_amp=False`` on Pascal (P100)
   or CPU-only runs.

2. **Non-blocking H→D transfers** — all ``.to(device)`` calls use
   ``non_blocking=True`` so the CPU-to-GPU copy overlaps with the previous
   GPU kernel.  The pinned memory allocated by the DataLoader (``pin_memory=True``
   in train_large.py) makes this effective.

3. **``zero_grad(set_to_none=True)``** — releases gradient tensors from GPU
   memory instead of zeroing them in-place, reducing peak VRAM and saving a
   memset kernel launch each step.

4. **Combined dtype+device cast** — ``.to(device).float()`` is replaced by a
   single ``.to(device, dtype=torch.float32, non_blocking=True)`` call, which
   avoids a superfluous intermediate tensor.

5. **Epoch-level loss accumulation** — ``train_loss_sum`` accumulates the
   Python float returned by ``.item()``; since ``.backward()`` already acts as
   an implicit sync barrier before the next forward pass, calling ``.item()``
   immediately after does not add meaningful extra latency.

Usage
-----
    from mySRForGAN import MySRForGAN

    model = MySRForGAN(scoring_rule='energy', n_samples_sr=10, use_amp=True)
    model.set_generator(condition_size=22, output_dim=1,
                        hidden_dim_rnn=64, n_layers=2, rnn_layer='lstm')
    model.train(train_dataset, val_data=val_dataset,
                checkpoint_dir='./checkpoints', checkpoint_every=1,
                resume_from='./checkpoints/checkpoint_epoch_0005.pt',
                max_runtime_seconds=23.5 * 3600,
                time_save_buffer_seconds=300)
"""

import copy
import os
import time as _time
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

    GPU note
    --------
    All tensors stay on whatever device ``samples`` lives on.
    ``torch.eye`` is created on the same device to avoid an implicit H→D copy.
    The (batch, m, m) pairwise distance matrix stays on GPU throughout.
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

    # Mask diagonal to exclude j==k pairs — created on the correct device
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
    z_noise_dim        : int    — latent noise dimension
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
    use_amp            : bool   — enable Automatic Mixed Precision (float16 matmuls).
                                  Defaults to True when CUDA is available.
                                  Set False on CPU, MPS, or old Pascal GPUs (P100).
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
        use_amp: bool = True,
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
        self.early_stopped = False

        # AMP: only meaningful on CUDA; silently disabled on CPU/MPS
        self.use_amp = use_amp and (self.DEVICE.type == "cuda")
        if use_amp and not self.use_amp:
            import warnings
            warnings.warn(
                "use_amp=True requested but CUDA is not available. "
                "AMP will be disabled.",
                UserWarning,
            )

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

        GPU note
        --------
        ``z`` is created directly on ``c.device`` and inherits the default
        dtype (float32).  Under AMP autocast, PyTorch automatically selects
        float16 for supported ops (Linear, LSTM) — no explicit cast needed here.
        """
        batch_size = c.size(0)
        m = self.n_samples_sr

        # Expand condition: (batch*m, condition_dim) — contiguous for LSTM efficiency
        c_expanded = c.unsqueeze(1).expand(-1, m, -1).reshape(batch_size * m, -1)

        # Independent noise draws — created on GPU directly (no H→D copy)
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

        GPU note
        --------
        Called inside ``torch.autocast`` context in ``train()``, so float16
        matmuls are used automatically when use_amp=True.  energy_score /
        kernel_score only use ``torch.norm`` and element-wise ops which are
        safe in float16 for our value ranges.
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
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        next_epoch: int,
        G_opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        best_val_loss: float,
        patience_counter: int,
        history: list,
    ) -> str:
        """
        Persist full training state to a .pt file.

        The filename encodes the *next* epoch to run so that the launcher
        script can pass it directly to ``resume_from`` without any arithmetic.

        Returns
        -------
        path : str  — absolute path of the written checkpoint file.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{next_epoch:05d}.pt"
        )
        torch.save(
            {
                # Training position: next_epoch is the *first* epoch that has
                # NOT been completed, so the resumed loop starts from here.
                "next_epoch":       next_epoch,
                # Model + optimiser state
                "G_state_dict":     self.G.state_dict(),
                "G_opt_state_dict": G_opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                # Early-stopping state
                "best_val_loss":    best_val_loss,
                "patience_counter": patience_counter,
                # Full history so far (list of dicts)
                "history":          history,
                # Metadata — useful for diagnostics
                "model_name":       self.MODEL_NAME,
                "max_epoch":        self.max_epoch,
                "scoring_rule":     self.scoring_rule,
            },
            path,
        )
        return path

    def _load_checkpoint(
        self,
        resume_from: str,
        G_opt: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> tuple:
        """
        Restore training state from a checkpoint file.

        Returns
        -------
        (start_epoch, best_val_loss, patience_counter, history)
        """
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(
                f"Checkpoint file not found: {resume_from}\n"
                "Check that $SCRATCH is set correctly and points to the same "
                "filesystem as the previous job."
            )

        # Always map to the device this process owns so multi-GPU jobs
        # land weights on the right card after a SLURM re-queue.
        ckpt = torch.load(resume_from, map_location=self.DEVICE)

        self.G.load_state_dict(ckpt["G_state_dict"])
        G_opt.load_state_dict(ckpt["G_opt_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch     = int(ckpt["next_epoch"])
        best_val_loss   = float(ckpt.get("best_val_loss", float("inf")))
        patience_counter = int(ckpt.get("patience_counter", 0))
        history         = list(ckpt.get("history", []))

        print(
            f"  Checkpoint loaded: {resume_from}\n"
            f"  Resuming from epoch {start_epoch}  "
            f"(best val SR so far: {best_val_loss:.6f}, "
            f"patience counter: {patience_counter})"
        )
        return start_epoch, best_val_loss, patience_counter, history

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(                                              # type: ignore[override]
        self,
        data: TensorDataset,
        val_data: Optional[TensorDataset] = None,
        save_history: bool = False,
        # ── Checkpoint / HPC job-chaining ────────────────────────────────────
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
        resume_from: Optional[str] = None,
        max_runtime_seconds: Optional[float] = None,
        time_save_buffer_seconds: float = 300.0,
    ):
        """
        Train the generator via scoring rule minimisation.

        The discriminator (self.D) is **not used**.  Only the generator (self.G)
        is optimised with Adam — no adversarial loop, no gradient penalty.

        Parameters
        ----------
        data         : TensorDataset of (y, c) pairs — training set
        val_data     : optional validation TensorDataset for early stopping
                       (strongly recommended: the SR loss is a meaningful metric,
                       unlike the GAN generator loss).
        save_history : bool — if True, saves per-epoch losses to CSV

        checkpoint_dir : str or None
            Directory for checkpoint files.  If None, no checkpoints are written.
            Checkpoints are named  checkpoint_epoch_{NNNNN}.pt  where NNNNN is
            the *next* epoch to run (so resuming is trivial).
            Use the same directory across jobs so the launcher can auto-detect
            the latest checkpoint.

        checkpoint_every : int (default 1)
            Save a checkpoint after every N completed epochs.
            1 = save every epoch (safest for short epoch times).
            5-10 = reasonable for long epochs where I/O matters.

        resume_from : str or None
            Path to a .pt checkpoint from a previous job.  All training state
            (weights, optimiser, scaler, early-stopping counters, history) is
            restored before the epoch loop starts.
            Pass None (default) to start training from scratch.

        max_runtime_seconds : float or None
            Maximum wall-clock seconds this job is allowed to run.
            Typically set to  (SLURM wall-time limit − buffer).
            Example: 23.5 * 3600 for a 24-hour job with 30 min buffer.
            When elapsed + time_save_buffer_seconds >= max_runtime_seconds
            the loop saves a checkpoint and returns early so the next SLURM
            job can resume.

        time_save_buffer_seconds : float (default 300 = 5 minutes)
            Minimum time reserved at the end of the job for writing the
            checkpoint.  Only relevant when max_runtime_seconds is set.

        GPU / HPC notes
        ---------------
        * The DataLoader is created by the caller's monkey-patch in train_large.py
          which injects ``num_workers``, ``pin_memory=True``, and
          ``persistent_workers=True`` for efficient async disk→GPU streaming.
        * ``non_blocking=True`` on the H→D transfers overlaps the PCIe copy with
          the previous kernel launch when pin_memory is active.
        * AMP GradScaler prevents float16 underflow in the backward pass;
          it is a no-op (identity) when use_amp=False so branching is unnecessary.
        """
        if self.G is None:
            raise RuntimeError(
                "Generator not defined. Call set_generator() first."
            )
        if not isinstance(data, torch.utils.data.Dataset):
            raise TypeError(
                f"data must be a torch.utils.data.Dataset. Got {type(data)}."
            )

        self.G.to(self.DEVICE)

        # DataLoaders are created here; train_large.py's monkey-patch injects
        # num_workers / pin_memory / persistent_workers transparently.
        train_loader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = (
            DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
            if val_data is not None
            else None
        )

        # Adam — plain (no momentum tuning needed for SR training)
        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.lr_g)

        # AMP GradScaler — no-op (enabled=False) when use_amp is False,
        # so the training loop below needs no branching.
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        amp_device_type = "cuda" if self.use_amp else "cpu"

        if self.use_amp:
            print(f"  AMP enabled — float16 matmuls active on {self.DEVICE}")

        # ── Early stopping + history state ───────────────────────────────────
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None
        history = []
        start_epoch = 0

        # ── Restore checkpoint if requested ──────────────────────────────────
        if resume_from is not None:
            start_epoch, best_val_loss, patience_counter, history = \
                self._load_checkpoint(resume_from, G_opt, scaler)

        # Restore best_state from the history to keep early-stopping coherent.
        # If we resumed and the model weights are already the "best" we
        # captured, take a snapshot now so early-stopping can still roll back.
        best_state = copy.deepcopy(self.G.state_dict())

        # ── Wall-clock timer ─────────────────────────────────────────────────
        job_start_time = _time.time()

        print(
            f"\n{'─'*60}\n"
            f"  Training {'resumed' if start_epoch > 0 else 'started'} "
            f"at epoch {start_epoch}/{self.max_epoch}\n"
            f"  checkpoint_dir  : {checkpoint_dir or 'disabled'}\n"
            f"  checkpoint_every: {checkpoint_every}\n"
            f"  max_runtime     : "
            + (f"{max_runtime_seconds/3600:.2f} h" if max_runtime_seconds else "unlimited")
            + f"\n{'─'*60}"
        )

        elapsed_at_exit = 0.0

        for epoch in range(start_epoch, self.max_epoch):

            # ── Wall-clock pre-check ─────────────────────────────────────────
            # Check BEFORE the epoch so we don't start work we can't finish.
            if max_runtime_seconds is not None:
                elapsed = _time.time() - job_start_time
                remaining = max_runtime_seconds - elapsed
                if remaining <= time_save_buffer_seconds:
                    print(
                        f"\n[TIME LIMIT] {elapsed/3600:.2f} h elapsed; "
                        f"only {remaining/60:.1f} min remaining "
                        f"(buffer={time_save_buffer_seconds/60:.0f} min).\n"
                        f"Saving checkpoint before epoch {epoch} and exiting."
                    )
                    if checkpoint_dir is not None:
                        ckpt_path = self._save_checkpoint(
                            checkpoint_dir, epoch, G_opt, scaler,
                            best_val_loss, patience_counter, history,
                        )
                        print(f"  Checkpoint saved → {ckpt_path}")
                    elapsed_at_exit = elapsed
                    break

            # ── Training pass ────────────────────────────────────────────────
            self.G.train()
            train_loss_sum = 0.0

            for y_batch, c_batch in train_loader:
                # Combined dtype + device cast in one operation.
                # non_blocking=True overlaps the PCIe DMA with the previous GPU
                # kernel when the source tensor lives in pinned (page-locked) memory,
                # which is guaranteed by pin_memory=True in the DataLoader.
                y_batch = y_batch.to(self.DEVICE, dtype=torch.float32, non_blocking=True)
                c_batch = c_batch.to(self.DEVICE, dtype=torch.float32, non_blocking=True)

                # set_to_none=True frees gradient memory instead of zeroing it:
                # fewer bytes written, one less memset kernel launch per step.
                G_opt.zero_grad(set_to_none=True)

                # autocast selects float16 for supported ops (Linear, LSTM, etc.)
                # and keeps float32 for ops that need it (norms, exp, softmax).
                # When use_amp=False, autocast is a transparent no-op.
                with torch.autocast(device_type=amp_device_type,
                                    dtype=torch.float16,
                                    enabled=self.use_amp):
                    loss = self._compute_sr_loss(c_batch, y_batch)

                # scaler.scale(loss).backward() is equivalent to loss.backward()
                # when use_amp=False (scaler is identity in that case).
                scaler.scale(loss).backward()
                scaler.step(G_opt)
                scaler.update()

                train_loss_sum += loss.item()

            avg_train = train_loss_sum / len(train_loader)

            # ── Validation pass ──────────────────────────────────────────────
            avg_val: Optional[float] = None
            if val_loader is not None:
                self.G.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for y_v, c_v in val_loader:
                        y_v = y_v.to(self.DEVICE, dtype=torch.float32, non_blocking=True)
                        c_v = c_v.to(self.DEVICE, dtype=torch.float32, non_blocking=True)
                        # AMP in validation: reduces VRAM and speeds up inference.
                        # torch.no_grad() + autocast is the standard pattern.
                        with torch.autocast(device_type=amp_device_type,
                                            dtype=torch.float16,
                                            enabled=self.use_amp):
                            val_loss_sum += self._compute_sr_loss(c_v, y_v).item()
                avg_val = val_loss_sum / len(val_loader)

            # ── Logging ──────────────────────────────────────────────────────
            elapsed_now = _time.time() - job_start_time
            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                val_str = f"  Val SR: {avg_val:.6f}" if avg_val is not None else ""
                print(
                    f"Epoch {epoch:4d}/{self.max_epoch}  "
                    f"Train SR: {avg_train:.6f}{val_str}  "
                    f"[{elapsed_now/3600:.2f} h]"
                )

            history.append(
                {"epoch": epoch, "train_sr": avg_train, "val_sr": avg_val}
            )

            # ── Early stopping ───────────────────────────────────────────────
            if avg_val is not None:
                if avg_val < best_val_loss - self.early_stopping_min_delta:
                    best_val_loss = avg_val
                    patience_counter = 0
                    # state_dict() lives on GPU but is a plain dict of tensors —
                    # deepcopy correctly clones them on the same device.
                    best_state = copy.deepcopy(self.G.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(
                            f"\nEarly stopping at epoch {epoch}. "
                            f"Best val SR: {best_val_loss:.6f}"
                        )
                        self.early_stopped = True
                        if best_state is not None:
                            self.G.load_state_dict(best_state)
                        # Save one final checkpoint before exiting early.
                        if checkpoint_dir is not None:
                            ckpt_path = self._save_checkpoint(
                                checkpoint_dir, epoch + 1, G_opt, scaler,
                                best_val_loss, patience_counter, history,
                            )
                            print(f"  Early-stop checkpoint saved → {ckpt_path}")
                        break

            # ── Periodic checkpoint ──────────────────────────────────────────
            if (
                checkpoint_dir is not None
                and (epoch + 1) % checkpoint_every == 0
            ):
                ckpt_path = self._save_checkpoint(
                    checkpoint_dir, epoch + 1, G_opt, scaler,
                    best_val_loss, patience_counter, history,
                )
                print(f"  [ckpt] Saved → {ckpt_path}")

        # ── Post-loop wall-clock check ────────────────────────────────────────
        # If the loop exited normally (completed all epochs or early stopping)
        # but we haven't saved a final checkpoint yet, do so now.
        else:
            # ``else`` on a for-loop runs only when it exits without ``break``
            elapsed_at_exit = _time.time() - job_start_time
            if checkpoint_dir is not None:
                final_ckpt = self._save_checkpoint(
                    checkpoint_dir, self.max_epoch, G_opt, scaler,
                    best_val_loss, patience_counter, history,
                )
                print(f"  [ckpt] Final checkpoint saved → {final_ckpt}")

        total_elapsed = _time.time() - job_start_time
        print(f"\nTraining finished in {total_elapsed:.1f}s ({total_elapsed/3600:.2f} h).")

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

        GPU note
        --------
        ``G.save()`` calls ``torch.save(state_dict, path)``.  PyTorch serialises
        GPU tensors correctly; load with ``map_location=device`` to control
        which GPU (or CPU) the weights land on.
        """
        import json
        os.makedirs(save_dir, exist_ok=True)

        gen_path = os.path.join(save_dir, f"{self.MODEL_NAME}_generator.pth")
        self.save_generator(gen_path)

        cfg = {
            "max_epoch":        self.max_epoch,
            "batch_size":       self.batch_size,
            "z_dim":            self.z_dim,
            "n_samples_sr":     self.n_samples_sr,
            "scoring_rule":     self.scoring_rule,
            "beta":             self.beta,
            "kernel_bandwidth": self.kernel_bandwidth,
            "model_name":       self.MODEL_NAME,
            "lr_g":             self.lr_g,
            "use_amp":          self.use_amp,
        }
        cfg_path = os.path.join(save_dir, f"{self.MODEL_NAME}_config.json")
        with open(cfg_path, "w") as f:
            import json
            json.dump(cfg, f, indent=2)
        print(f"SR config saved to {cfg_path}")

    def load_models(self, load_dir: str = "./models"):
        """
        Load the generator (and config) from *load_dir*.
        No discriminator is loaded.

        GPU note
        --------
        ``MyGenerator.load()`` calls ``torch.load(map_location=device)`` so
        weights always land on the device detected at construction time.
        """
        import json
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
            self.use_amp          = cfg.get("use_amp",          self.use_amp)
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
        """
        loader = DataLoader(data, batch_size=n_pairs, shuffle=True)
        y_batch, _ = next(iter(loader))
        y_np = y_batch.numpy().reshape(len(y_batch), -1)

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