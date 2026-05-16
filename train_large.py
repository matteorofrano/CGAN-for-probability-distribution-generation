"""
train_large.py
==============
End-to-end SRForGAN training pipeline on large Heston stochastic-volatility
datasets (e.g. 1 million paths) that exceed available RAM.

Model
-----
MySRForGAN — a scoring-rule generator (Energy Score) that needs NO discriminator.
The generator optionally uses an LSTM RNN encoder (RnnGenerator) instead of a
plain MLP (MyGenerator), controlled by ``USE_LSTM``.

Data pattern (SRForGAN)
-----------------------
For each simulated trajectory  [X_0, X_1, …, X_N]  (shape N+1):

    condition  c = trajectory[:, :-1]   shape (J, N)   ← look-back window
    target     y = trajectory[:, -1:]   shape (J, 1)   ← 1-step-ahead log-price

Workflow
--------
1. GENERATE  – Simulate Heston paths in chunks; append each chunk's
               trajectories (J × (N+1) float32) to one flat binary file.
               A companion JSON header records the record length so the
               reader can seek directly to any record without loading the
               whole file.

2. STREAM    – HestonBinaryDataset memory-maps the binary file and returns
               (target, condition) tensors on demand; the full dataset
               never sits in RAM simultaneously.
               *** The memmap is opened lazily inside __getitem__ so that
               the object is safely picklable across DataLoader worker
               processes (avoids "cannot pickle mmap" errors). ***

3. TUNE      – Optional Optuna hyperparameter search (set RUN_TUNING=True).
               Disabled by default.

4. TRAIN     – Train MySRForGAN (with or without LSTM) on the full streamed
               dataset using the Energy Score loss.

5. SAVE      – Generator weights + config JSON + training config JSON.

Usage
-----
    python train_large.py

    # Optional environment variables:
    #   CUDA_VISIBLE_DEVICES=0
    #   OMP_NUM_THREADS=4
    # Increase NUM_WORKERS on nodes with fast NVMe storage (e.g. 4–8).
    # Set NUM_WORKERS=0 if you hit "cannot pickle" or "too many open files".
"""

from __future__ import annotations

import json
import os
import struct
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from heston_data_simulator import HestonSimulator
from mySRForGAN import MySRForGAN


# ── Configuration ──────────────────────────────────────────────────────────────

# --- Heston parameter ranges (SPX-like) ---
X0_RANGE      = (0.0,  0.0)      # initial log-price (fixed at 0 = normalised)
MU_RANGE      = (0.0,  0.0)     # risk-neutral drift
V0_RANGE      = (0.6, 0.6)     # initial variance 
KAPPA_RANGE   = (1.0,  1.0)      # mean-reversion speed
THETA_RANGE   = (0.6, 0.6)     # long-run variance
SIGMA_V_RANGE = (0.5,  0.5)      # vol-of-vol
RHO_RANGE     = (0.7, 0.7)     # spot-vol correlation

# --- Time grid ---
N_STEPS = 22                     # number of time steps  
T       = round(22 / 252, 3)    # time horizon in years  

# --- Data volume ---
TOTAL_ROWS = 10_000_000
CHUNK_SIZE =   100_000           # paths per simulation batch (100 chunks of 100 K)
N_CHUNKS   = TOTAL_ROWS // CHUNK_SIZE

# --- File paths ---
BASE_DIR = os.environ.get("SCRATCH", ".")

DATA_DIR  = os.path.join(BASE_DIR, "srforgan_data")
MODEL_DIR = os.path.join(BASE_DIR, "srforgan_models")
BIN_FILE    = os.path.join(DATA_DIR, "heston_1M.bin")
HEADER_FILE = os.path.join(DATA_DIR, "heston_1M_header.json")


# --- Model architecture ---
USE_LSTM         = True          # True → RnnGenerator (LSTM); False → MLP generator
Z_NOISE_DIM      = 32            # latent noise dimension

# LSTM generator settings (used when USE_LSTM=True)
LSTM_HIDDEN_DIM  = 64            # hidden state size of the LSTM encoder
LSTM_N_LAYERS    = 1             # number of stacked LSTM layers
LSTM_DROPOUT     = 0.1           # inter-layer dropout (only active when N_LAYERS > 1)

# MLP generator settings (used when USE_LSTM=False)
MLP_HIDDEN_DIMS  = [128, 256, 128, 64]

# --- Training ---
MAX_EPOCH        = 100
BATCH_SIZE       = 128
LR_G             = 1e-3
N_SAMPLES_SR     = 10            # generator draws per Energy Score estimate
                                 # Paper uses 10; ≥3 suffices (Appendix F.3.2)
SCORING_RULE     = "energy"      # 'energy' | 'kernel' | 'energy_kernel'
EARLY_STOP_PAT   = 10            # epochs without improvement before stopping
VAL_FRACTION     = 0.2          # fraction of dataset used for validation
NUM_WORKERS      = 0             # DataLoader workers; 0 = main thread (safest).
                                 # Increase (e.g. 4) on HPC with fast NVMe storage.
                                 # Keep 0 on Windows or if "cannot pickle" errors occur.

# --- Optional hyperparameter tuning (disabled by default) ---
RUN_TUNING = False
N_TRIALS   = 30
N_SPLITS   = 3
CV_EPOCHS  = 40


# ── Binary file helpers ────────────────────────────────────────────────────────

RECORD_LEN = N_STEPS + 1        # (N+1) float32 values per record: X_0 … X_N
DTYPE      = np.float32


def _init_binary_file(bin_path: str, header_path: str, record_len: int) -> None:
    """
    Create (or truncate) the binary data file and write a fresh JSON header.

    Parameters
    ----------
    bin_path    : path to the .bin data file
    header_path : path to the companion JSON header file
    record_len  : number of float32 values per record (= N_STEPS + 1)
    """
    header = {"record_len": record_len, "n_records": 0}
    with open(header_path, "w") as f:
        json.dump(header, f)
    # Touch / truncate the binary file
    open(bin_path, "wb").close()


def _append_chunk_to_binary(
    bin_path: str, header_path: str, trajectories: np.ndarray
) -> None:
    """
    Append a chunk of trajectories (J, N+1) to the flat binary file and
    increment the record count in the header.

    Each record is stored as (N+1) contiguous float32 values.

    Parameters
    ----------
    bin_path     : path to the .bin data file
    header_path  : path to the companion JSON header file
    trajectories : np.ndarray shape (J, N+1) — log-price paths from the simulator
    """
    data = trajectories.astype(DTYPE)          # ensure float32; simulator returns float64
    with open(bin_path, "ab") as f:
        f.write(data.tobytes())                # raw bytes appended to end of file

    # Update record count in JSON header (read-modify-write)
    with open(header_path, "r") as f:
        header = json.load(f)
    header["n_records"] += len(trajectories)
    with open(header_path, "w") as f:
        json.dump(header, f)


# ── Streaming Dataset ──────────────────────────────────────────────────────────

class HestonBinaryDataset(Dataset):
    """
    Memory-mapped dataset for flat float32 trajectory files produced by
    ``_append_chunk_to_binary``.

    Data pattern (SRForGAN)
    -----------------------
    Each stored record is a full trajectory [X_0, X_1, …, X_N] of length N+1.
    ``__getitem__`` splits it into:

        condition : tensor shape (N_STEPS,)  — X_0 … X_{N-1}  (look-back window)
        target    : tensor shape (1,)        — X_N             (1-step-ahead log-price)

    returned as the pair  (target, condition) = (y, c)  expected by
    ``MySRForGAN.train()``.

    Lazy memmap
    -----------
    The ``np.memmap`` is opened inside ``__getitem__`` the first time it is
    needed (per-worker, not in ``__init__``).  This makes the dataset safely
    picklable for ``DataLoader(num_workers > 0)`` — numpy memmaps cannot be
    pickled across processes when created in ``__init__``.

    Parameters
    ----------
    bin_filepath    : path to the .bin data file
    header_filepath : path to the companion JSON header
    """

    def __init__(self, bin_filepath: str, header_filepath: str):
        with open(header_filepath) as f:
            header = json.load(f)

        self._record_len  = int(header["record_len"])   # N+1
        self._n_records   = int(header["n_records"])
        self._filepath    = bin_filepath
        self._mmap        = None                        # opened lazily in __getitem__

    def _open_mmap(self) -> None:
        """Open the memory-mapped file (called once per worker on first access)."""
        self._mmap = np.memmap(
            self._filepath,
            dtype=DTYPE,
            mode="r",
            shape=(self._n_records, self._record_len),
        )

    def __len__(self) -> int:
        return self._n_records

    def __getitem__(self, idx: int):
        # Lazy open: each DataLoader worker opens its own file handle on first access.
        if self._mmap is None:
            self._open_mmap()

        row = self._mmap[idx]                           # (N+1,) float32
        # SRForGAN data pattern:
        #   condition c = trajectory[:-1]   (N_STEPS,)  ← look-back window
        #   target    y = trajectory[-1:]   (1,)         ← 1-step-ahead log-price
        condition = torch.from_numpy(row[:-1].copy())   # (N_STEPS,)
        target    = torch.from_numpy(row[-1:].copy())   # (1,)
        return target, condition                        # (y, c) — MySRForGAN format

    def __getstate__(self):
        """Custom pickle: drop the mmap so the object is safely serialisable."""
        state = self.__dict__.copy()
        state["_mmap"] = None                           # will be re-opened by worker
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ── Step 1: Generate data in chunks ───────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Skip generation if a complete binary file already exists.
# This lets you resume a crashed run or reuse a previously generated
# dataset without re-simulating 1 M paths from scratch.
# ------------------------------------------------------------------
_regen = True
if os.path.exists(BIN_FILE) and os.path.exists(HEADER_FILE):
    try:
        with open(HEADER_FILE) as _f:
            _hdr = json.load(_f)
        if int(_hdr.get("n_records", 0)) == TOTAL_ROWS:
            print(
                f"Binary file already contains {TOTAL_ROWS:,} records — "
                "skipping data generation.\n"
                f"  (Delete '{BIN_FILE}' to regenerate.)\n"
            )
            _regen = False
    except (json.JSONDecodeError, KeyError):
        pass  # corrupted header → regenerate

if _regen:
    print(f"{'='*60}")
    print(f"  Generating {TOTAL_ROWS:,} Heston paths")
    print(f"  {N_CHUNKS} chunks × {CHUNK_SIZE:,} paths each")
    print(f"  N={N_STEPS} steps, T={T} yr  (dt = 1/252 per step)")
    print(f"{'='*60}\n")

    _init_binary_file(BIN_FILE, HEADER_FILE, RECORD_LEN)

    for i in range(N_CHUNKS):
        sim = HestonSimulator(
            X0_range      = X0_RANGE,
            mu_range      = MU_RANGE,
            v0_range      = V0_RANGE,
            kappa_range   = KAPPA_RANGE,
            theta_range   = THETA_RANGE,
            sigma_v_range = SIGMA_V_RANGE,
            rho_range     = RHO_RANGE,
            T             = T,
            N             = N_STEPS,
            n_simulations = CHUNK_SIZE,
            seed          = 42 + i,       # independent, reproducible seed per chunk
            scheme        = "milstein",   # full-truncation Milstein for variance SDE
        )

        # get_paths() returns full log-price trajectories: shape (CHUNK_SIZE, N_STEPS+1)
        #   trajectories[:, :-1]  → condition window  (N_STEPS columns)
        #   trajectories[:, -1:]  → 1-step-ahead target (1 column)
        # Both are derived on-the-fly in HestonBinaryDataset.__getitem__; we only
        # store the full (N+1)-step trajectory in the binary file.
        trajectories = sim.get_paths()    # np.ndarray (CHUNK_SIZE, N_STEPS+1), float64

        _append_chunk_to_binary(BIN_FILE, HEADER_FILE, trajectories)

        total_so_far = (i + 1) * CHUNK_SIZE
        pct          = 100.0 * (i + 1) / N_CHUNKS
        print(
            f"  Chunk {i + 1:2d}/{N_CHUNKS}  appended "
            f"({pct:5.1f} %)  —  {total_so_far:,} rows total"
        )

    print(f"\nData generation complete.  File: {BIN_FILE}\n")


# ── Step 2: Build streaming dataset ───────────────────────────────────────────

full_dataset = HestonBinaryDataset(BIN_FILE, HEADER_FILE)

# Sanity: read the header to confirm sizes before training
with open(HEADER_FILE) as _f:
    _hdr = json.load(_f)
assert int(_hdr["n_records"]) == TOTAL_ROWS, (
    f"Expected {TOTAL_ROWS:,} records in binary file; "
    f"found {_hdr['n_records']:,}.  Delete '{BIN_FILE}' and re-run."
)

print(f"HestonBinaryDataset ready:")
print(f"  Total records  : {len(full_dataset):,}")
print(f"  Record length  : {full_dataset._record_len}  (= N_STEPS + 1 = {N_STEPS + 1})")
print(f"  Condition size : {full_dataset._record_len - 1}  (= N_STEPS = {N_STEPS})")
print(f"  Target size    : 1  (next log-price X_{{N}})\n")

# --- Train / validation split ---
n_val   = int(len(full_dataset) * VAL_FRACTION)
n_train = len(full_dataset) - n_val
train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(0),
)
print(f"  Train split    : {n_train:,}  ({100 * (1 - VAL_FRACTION):.0f} %)")
print(f"  Val split      : {n_val:,}   ({100 * VAL_FRACTION:.0f} %)\n")


# ── Step 3 (Optional): Hyperparameter tuning ──────────────────────────────────

if RUN_TUNING:
    print(f"{'='*60}")
    print(f"  Hyperparameter tuning  ({N_TRIALS} trials × {N_SPLITS}-fold CV)")
    print(f"{'='*60}\n")

    try:
        import optuna
        from sklearn.model_selection import KFold
    except ImportError:
        warnings.warn(
            "optuna or scikit-learn not installed — skipping hyperparameter tuning. "
            "Install with:  pip install optuna scikit-learn",
            RuntimeWarning,
        )
        RUN_TUNING = False

if RUN_TUNING:

    def _make_model_for_trial(trial: "optuna.Trial") -> MySRForGAN:
        """Build a MySRForGAN from an Optuna trial's suggested params."""
        use_lstm_trial = trial.suggest_categorical("use_lstm", [True, False])
        z_dim  = trial.suggest_categorical("z_noise_dim", [16, 32, 64])
        lr     = trial.suggest_float("lr_g", 1e-4, 5e-3, log=True)
        n_samp = trial.suggest_int("n_samples_sr", 5, 20)
        bs     = trial.suggest_categorical("batch_size", [256, 512, 1024])

        m = MySRForGAN(
            max_epoch               = CV_EPOCHS,
            batch_size              = bs,
            z_noise_dim             = z_dim,
            n_samples_sr            = n_samp,
            scoring_rule            = SCORING_RULE,
            lr_g                    = lr,
            early_stopping_patience = 10,
            name                    = f"srforgan_trial_{trial.number}",
        )

        if use_lstm_trial:
            hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128])
            n_layers   = trial.suggest_int("lstm_n_layers", 1, 3)
            m.set_generator(
                condition_size = N_STEPS,
                output_dim     = 1,
                hidden_dim_rnn = hidden_dim,   # triggers RnnGenerator
                n_layers       = n_layers,
                rnn_layer      = "lstm",
                dropout        = trial.suggest_float("dropout", 0.0, 0.3),
            )
        else:
            hidden_dims_str = trial.suggest_categorical(
                "mlp_hidden_dims",
                ["[64,128,64]", "[128,256,128]", "[128,256,256,128,64]"],
            )
            m.set_generator(
                condition_size = N_STEPS,
                output_dim     = 1,
                hidden_dims    = json.loads(hidden_dims_str),
                use_batch_norm = trial.suggest_categorical("use_bn", [True, False]),
            )
        return m

    def _objective(trial: "optuna.Trial") -> float:
        """K-fold CV objective: mean best validation Energy Score across folds."""
        kf      = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        indices = np.arange(n_train)
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(indices)):
            fold_train = torch.utils.data.Subset(train_dataset, tr_idx.tolist())
            fold_val   = torch.utils.data.Subset(train_dataset, va_idx.tolist())

            m       = _make_model_for_trial(trial)
            history = m.train(fold_train, val_data=fold_val, save_history=False)

            val_scores = [h["val_sr"] for h in history if h["val_sr"] is not None]
            fold_scores.append(min(val_scores) if val_scores else float("inf"))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction      = "minimize",
        study_name     = "srforgan_heston",
        storage        = "sqlite:///srforgan_study.db",
        load_if_exists = True,
    )
    study.optimize(_objective, n_trials=N_TRIALS)

    print(f"\nBest trial score : {study.best_value:.6f}")
    print(f"Best params      : {study.best_params}\n")

    # Override architecture settings with best params found
    best_p        = study.best_params
    USE_LSTM      = best_p.get("use_lstm",      USE_LSTM)
    Z_NOISE_DIM   = best_p.get("z_noise_dim",   Z_NOISE_DIM)
    LR_G          = best_p.get("lr_g",          LR_G)
    N_SAMPLES_SR  = best_p.get("n_samples_sr",  N_SAMPLES_SR)
    BATCH_SIZE    = best_p.get("batch_size",     BATCH_SIZE)
    if USE_LSTM:
        LSTM_HIDDEN_DIM = best_p.get("lstm_hidden_dim", LSTM_HIDDEN_DIM)
        LSTM_N_LAYERS   = best_p.get("lstm_n_layers",   LSTM_N_LAYERS)
        LSTM_DROPOUT    = best_p.get("dropout",          LSTM_DROPOUT)
    else:
        MLP_HIDDEN_DIMS = json.loads(
            best_p.get("mlp_hidden_dims", str(MLP_HIDDEN_DIMS))
        )


# ── Step 4: Build and train SRForGAN ──────────────────────────────────────────

print(f"{'='*60}")
arch_label = "LSTM  (RnnGenerator)" if USE_LSTM else "MLP   (MyGenerator)"
print(f"  Final training  —  {arch_label}")
print(f"  Dataset        : {n_train:,} train / {n_val:,} val")
print(f"  Max epochs     : {MAX_EPOCH}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  LR generator   : {LR_G}")
print(f"  Scoring rule   : {SCORING_RULE}")
print(f"  m draws / SR   : {N_SAMPLES_SR}")
print(f"{'='*60}\n")

model = MySRForGAN(
    max_epoch               = MAX_EPOCH,
    batch_size              = BATCH_SIZE,
    z_noise_dim             = Z_NOISE_DIM,
    n_samples_sr            = N_SAMPLES_SR,
    scoring_rule            = SCORING_RULE,
    lr_g                    = LR_G,
    early_stopping_patience = EARLY_STOP_PAT,
    name                    = "srforgan_heston",
)

if USE_LSTM:
    # ── RnnGenerator: LSTM encodes the condition window ─────────────────────
    # set_generator routes to RnnGenerator when hidden_dim_rnn is not None.
    #
    # condition_size = N_STEPS  → the LSTM sees N_STEPS univariate time steps.
    # output_dim     = 1        → scalar next log-price.
    # hidden_dim_rnn           → passed as `hidden_dim` to RnnGenerator.
    # n_layers / rnn_layer / dropout → forwarded via **kwargs to RnnGenerator.
    model.set_generator(
        condition_size = N_STEPS,
        output_dim     = 1,
        hidden_dim_rnn = LSTM_HIDDEN_DIM,   # triggers RnnGenerator branch
        n_layers       = LSTM_N_LAYERS,
        rnn_layer      = "lstm",
        dropout        = LSTM_DROPOUT,
    )
    print(
        f"Generator: RnnGenerator (LSTM)\n"
        f"  condition_size = {N_STEPS} (look-back window)\n"
        f"  hidden_dim     = {LSTM_HIDDEN_DIM}\n"
        f"  n_layers       = {LSTM_N_LAYERS}\n"
        f"  dropout        = {LSTM_DROPOUT}\n"
        f"  z_noise_dim    = {Z_NOISE_DIM}\n"
    )
else:
    # ── MyGenerator: plain MLP — condition and noise are concatenated ────────
    model.set_generator(
        condition_size = N_STEPS,
        output_dim     = 1,
        hidden_dims    = MLP_HIDDEN_DIMS,
        use_batch_norm = True,
    )
    print(
        f"Generator: MyGenerator (MLP)\n"
        f"  condition_size = {N_STEPS} (look-back window)\n"
        f"  hidden_dims    = {MLP_HIDDEN_DIMS}\n"
        f"  z_noise_dim    = {Z_NOISE_DIM}\n"
    )

total_params = sum(p.numel() for p in model.G.parameters() if p.requires_grad)
print(f"Trainable parameters : {total_params:,}\n")

# ── DataLoader configuration ───────────────────────────────────────────────────
# MySRForGAN.train() creates DataLoaders internally without worker settings.
# We temporarily replace torch.utils.data.DataLoader with a thin wrapper that
# injects num_workers / pin_memory / persistent_workers for efficient disk
# streaming from the memory-mapped binary file.
# The original class is restored immediately after training.

_OrigDataLoader = torch.utils.data.DataLoader

def _streaming_dataloader(ds, **kw):
    if NUM_WORKERS > 0:
        kw.setdefault("num_workers",        NUM_WORKERS)
        kw.setdefault("pin_memory",         True)
        kw.setdefault("persistent_workers", True)
        kw.setdefault("prefetch_factor",    2)
    return _OrigDataLoader(ds, **kw)

torch.utils.data.DataLoader = _streaming_dataloader  # patch

history = model.train(
    data         = train_dataset,
    val_data     = val_dataset,
    save_history = False,           
)

torch.utils.data.DataLoader = _OrigDataLoader         # restore


# ── Step 5: Save ──────────────────────────────────────────────────────────────

model.save_models(MODEL_DIR)

# Also save the training configuration alongside the model weights so that
# the experiment is fully reproducible.
training_cfg = {
    # Heston simulation
    "X0_range":      list(X0_RANGE),
    "mu_range":      list(MU_RANGE),
    "v0_range":      list(V0_RANGE),
    "kappa_range":   list(KAPPA_RANGE),
    "theta_range":   list(THETA_RANGE),
    "sigma_v_range": list(SIGMA_V_RANGE),
    "rho_range":     list(RHO_RANGE),
    "N_steps":       N_STEPS,
    "T":             T,
    "total_rows":    TOTAL_ROWS,
    "chunk_size":    CHUNK_SIZE,
    # Architecture
    "use_lstm":      USE_LSTM,
    "z_noise_dim":   Z_NOISE_DIM,
    "lstm_hidden_dim": LSTM_HIDDEN_DIM if USE_LSTM else None,
    "lstm_n_layers":   LSTM_N_LAYERS   if USE_LSTM else None,
    "lstm_dropout":    LSTM_DROPOUT    if USE_LSTM else None,
    "mlp_hidden_dims": MLP_HIDDEN_DIMS if not USE_LSTM else None,
    # Training
    "max_epoch":       MAX_EPOCH,
    "batch_size":      BATCH_SIZE,
    "lr_g":            LR_G,
    "n_samples_sr":    N_SAMPLES_SR,
    "scoring_rule":    SCORING_RULE,
    "early_stop_pat":  EARLY_STOP_PAT,
    "val_fraction":    VAL_FRACTION,
}
train_cfg_path = os.path.join(MODEL_DIR, "srforgan_heston_training_config.json")
with open(train_cfg_path, "w") as _f:
    json.dump(training_cfg, _f, indent=2)
print(f"Training config saved to {train_cfg_path}")
print(f"\nDone.  Models saved to ./{MODEL_DIR}/srforgan_heston_*\n")


# ── Step 6: Sanity check — sample from the trained generator ──────────────────

print("--- Sanity check: draw samples from trained generator ---")
device = model.DEVICE
model.G.eval()

# Grab one validation batch — DataLoader is created directly here (no patch needed)
_val_loader = _OrigDataLoader(val_dataset, batch_size=8, shuffle=True)
y_batch, c_batch = next(iter(_val_loader))
y_batch = y_batch.to(device).float()   # (8, 1)
c_batch = c_batch.to(device).float()   # (8, N_STEPS)

with torch.no_grad():
    m_draws = 200  # number of samples drawn from the conditional distribution

    # Replicate each condition m_draws times for parallel sampling
    # c_batch: (8, N_STEPS) → c_exp: (8*m_draws, N_STEPS)
    c_exp = c_batch.unsqueeze(1).expand(-1, m_draws, -1).reshape(8 * m_draws, -1)

    # Independent noise draws (reparametrisation trick)
    z = torch.randn(8 * m_draws, model.z_dim, device=device)

    # G(c, z) works for both MyGenerator and RnnGenerator
    fake = model.G(c_exp, z).view(8, m_draws, 1)   # (8, m_draws, 1)

# Report predictive mean MAE vs realised target
pred_mean = fake.mean(dim=1).cpu().numpy()   # (8, 1)
true_vals = y_batch.cpu().numpy()            # (8, 1)
mae       = float(np.abs(pred_mean - true_vals).mean())
print(f"  Predictive mean MAE (8 samples, m={m_draws} draws): {mae:.6f}")
print(
    "  NOTE: MAE ≈ 0 is not the training objective.\n"
    "        The goal is correct spread / calibration of the predictive distribution."
)