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
 
    condition  c = trajectory[:, :-1]   shape (J, N)   <- look-back window
    target     y = trajectory[:, -1:]   shape (J, 1)   <- 1-step-ahead log-price
 
Workflow
--------
1. GENERATE  -- Simulate Heston paths in chunks; append each chunk's
               trajectories (J x (N+1) float32) to one flat binary file.
2. STREAM    -- HestonBinaryDataset memory-maps the binary file.  The memmap
               is opened lazily inside __getitem__ so the object is safely
               picklable across DataLoader worker processes.
3. TUNE      -- Optional Optuna hyperparameter search (RUN_TUNING=True).
4. TRAIN     -- Train MySRForGAN (with or without LSTM + AMP) on the streamed
               dataset using the Energy Score loss.
5. SAVE      -- Generator weights + config JSON + training config JSON.
 
HPC / GPU notes
---------------
* SLURM environment variables are detected automatically.  On a multi-GPU node
  SLURM_LOCALID selects which GPU this rank owns; CUDA_VISIBLE_DEVICES can
  also be set externally (e.g. by the scheduler).
 
* torch.backends.cudnn.benchmark = True enables the cuDNN autotuner for
  fixed-size inputs (same batch size every step), giving ~10-30% extra speed.
 
* USE_AMP = True enables Automatic Mixed Precision in MySRForGAN -- float16
  matmuls on the GPU forward pass, GradScaler in backward.  Turn off on Pascal
  (P100) GPUs or any CPU-only run.
 
* NUM_WORKERS controls async data loading.  On HPC nodes with NVMe scratch
  disks, 4-8 workers keep the GPU fed.  Set 0 on Windows or shared filesystems
  with strict inode limits.
 
* pin_memory = True (injected via the DataLoader wrapper) is what makes
  non_blocking=True transfers inside mySRForGAN.train() effective -- the
  DataLoader allocates page-locked host memory so the PCIe DMA can overlap
  with GPU kernel execution.
 
Checkpoint / job-chaining design
---------------------------------
CINECA (and most HPC clusters) enforces a 24-hour wall-time limit per job.
Training 10 million Heston trajectories for 100+ epochs will exceed this.
 
The solution implemented here:
 
  1. Every completed epoch is checkpointed to  $SCRATCH/srforgan_checkpoints/
     (controlled by CHECKPOINT_EVERY).
 
  2. A time-limit guard inside mySRForGAN.train() monitors elapsed wall-clock
     time.  When  elapsed + TIME_SAVE_BUFFER_MINS  >= MAX_JOB_HOURS  the
     guard saves a checkpoint and returns early.
 
  3. At job start this script scans the checkpoint directory and automatically
     passes the latest .pt file to  resume_from.  No manual editing is needed
     between jobs.
 
  4. A companion SLURM array script (see Usage below) submits follow-up jobs
     automatically until all epochs are complete.
 
File paths and $SCRATCH
-----------------------
All large artefacts (binary data, checkpoints, models) are placed under
$SCRATCH, the high-performance parallel filesystem on CINECA clusters.
$SCRATCH is set by the cluster's environment module; if it is NOT set the
script falls back to the current working directory but prints a warning,
because local disk on a compute node is usually /tmp (small and ephemeral).
 
  ┌─────────────────────────────────────────────────────┐
  │  Always submit your job from a directory INSIDE     │
  │  $SCRATCH, or set SCRATCH explicitly in your        │
  │  SLURM script:                                      │
  │      export SCRATCH=/path/to/your/scratch           │
  └─────────────────────────────────────────────────────┘
 
Usage (HPC / SLURM)
--------------------
    # single-GPU job (also used for follow-up jobs — auto-resumes):
    python train_large.py
 
    # multi-GPU per node (one process per GPU, launched by SLURM):
    srun --ntasks-per-node=4 --gpus-per-task=1 python train_large.py
 
    # recommended SLURM script (job array for automatic chaining):
    #   See srforgan_array.sh generated at the bottom of this file.
 
"""
 
from __future__ import annotations
 
import glob
import json
import os
import sys
import warnings
from typing import Optional
 
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
 
from heston_data_simulator import HestonSimulator
from mySRForGAN import MySRForGAN


# ============================================================
# $SCRATCH path validation
# ============================================================
#
# On CINECA, $SCRATCH is set by the environment module loaded in the
# SLURM script (e.g. "module load profile/base").  If it is absent
# the code falls back to "." but warns loudly so the user notices.
#
# Common pitfall: submitting the job from $HOME instead of $SCRATCH
# causes all I/O to go to a slow NFS mount.  The check below catches
# this early.
 
_SCRATCH_RAW = os.environ.get("SCRATCH", "")
 
if _SCRATCH_RAW:
    # Resolve any symlinks (CINECA scratch is sometimes a symlink)
    BASE_DIR = os.path.realpath(_SCRATCH_RAW)
    if not os.path.isdir(BASE_DIR):
        sys.exit(
            f"[FATAL] $SCRATCH='{_SCRATCH_RAW}' resolved to '{BASE_DIR}' "
            "which does not exist.\n"
            "Check that the filesystem is mounted and try again."
        )
    print(f"$SCRATCH resolved to : {BASE_DIR}")
else:
    BASE_DIR = os.path.realpath(".")
    warnings.warn(
        "\n"
        "  $SCRATCH is not set — writing all files to the current working\n"
        "  directory: " + BASE_DIR + "\n"
        "  On a compute node this may be a small local disk (/tmp) or a\n"
        "  slow NFS mount.  Set SCRATCH in your SLURM script to avoid\n"
        "  running out of space or I/O bottlenecks.\n"
        "  Example: #SBATCH --export=ALL,SCRATCH=/path/to/scratch",
        RuntimeWarning,
        stacklevel=1,
    )
 
 
# ============================================================
# GPU / HPC environment setup
# ============================================================
 
# SLURM-aware GPU selection.
# When SLURM launches one process per GPU on a node it sets SLURM_LOCALID
# (0-indexed rank on this node).  We bind each process to its own GPU so
# all ranks stay off each other's VRAM.  Falls back to GPU 0 / CPU gracefully.
_local_rank = int(os.environ.get("SLURM_LOCALID", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(_local_rank)
    _device_str = f"cuda:{_local_rank}"
else:
    _device_str = "cpu"
 
print(f"Process bound to device : {_device_str}")
print(f"  torch version   : {torch.__version__}")
print(f"  CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(_local_rank)
    print(f"  GPU name        : {torch.cuda.get_device_name(_local_rank)}")
    print(f"  VRAM total      : {_props.total_memory / 1e9:.1f} GB")
    print(f"  SM count        : {_props.multi_processor_count}")
print()
 
# cuDNN autotuner: benchmarks several GEMM/conv kernels on the first batch and
# picks the fastest for the fixed input shape.  Free ~10-30% speedup when the
# batch size is constant (guaranteed here by drop_last=True).
# No effect when running on CPU.
torch.backends.cudnn.benchmark = True
 
 
# ============================================================
# CONFIGURATION
# ============================================================
 
# --- Heston parameter ranges (SPX-like) ---
X0_RANGE      = (0.0,  0.0)      # initial log-price (fixed at 0 = normalised)
MU_RANGE      = (0.0,  0.0)      # risk-neutral drift
V0_RANGE      = (0.6,  0.6)      # initial variance
KAPPA_RANGE   = (1.0,  1.0)      # mean-reversion speed
THETA_RANGE   = (0.6,  0.6)      # long-run variance
SIGMA_V_RANGE = (0.5,  0.5)      # vol-of-vol
RHO_RANGE     = (-0.7,  -0.7)      # spot-vol correlation
 
# --- Time grid ---
CONDITION_STEPS = 252
FORECAST_HORIZON = 5 # one week
N_TOTAL = CONDITION_STEPS + FORECAST_HORIZON                    # number of time steps
T       = N_TOTAL/252    # time horizon in years
 
# --- Data volume ---
TOTAL_ROWS = 10_000_000
CHUNK_SIZE =    100_000          # paths per simulation batch (100 chunks of 100 K)
N_CHUNKS   = TOTAL_ROWS // CHUNK_SIZE
 
# --- File paths (all under $SCRATCH) ---
DATA_DIR        = os.path.join(BASE_DIR, "srforgan_data")
MODEL_DIR       = os.path.join(BASE_DIR, "srforgan_models")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "srforgan_checkpoints")
 
BIN_FILE    = os.path.join(DATA_DIR, "heston_10M.bin")
HEADER_FILE = os.path.join(DATA_DIR, "heston_10M_header.json")
 
# --- Model architecture ---
USE_LSTM         = True          # True → RnnGenerator (LSTM); False → MLP generator
Z_NOISE_DIM      = 64            # latent noise dimension
 
# LSTM generator settings (used when USE_LSTM=True)
LSTM_HIDDEN_DIM  = 128            # hidden state size of the LSTM encoder
LSTM_N_LAYERS    = 1             # number of stacked LSTM layers
LSTM_DROPOUT     = 0.1           # inter-layer dropout (only active when N_LAYERS > 1)
 
# MLP generator settings (used when USE_LSTM=False)
MLP_HIDDEN_DIMS  = [128, 256, 128, 64]
 
# --- Training ---
MAX_EPOCH        = 100
BATCH_SIZE       = 128
LR_G             = 1e-3
N_SAMPLES_SR     = 10            # generator draws per Energy Score estimate
SCORING_RULE     = "energy"      # 'energy' | 'kernel' | 'energy_kernel'
EARLY_STOP_PAT   = 10            # epochs without improvement before stopping
VAL_FRACTION     = 0.2           # fraction of dataset used for validation
 
# --- GPU throughput settings ---
USE_AMP     = torch.cuda.is_available()   # float16 on CUDA, disabled on CPU
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
 
# --- Checkpoint / job time-limit settings ---
#
# CINECA Leonardo / Galileo max wall-time: 24 h.
# We stop training MAX_JOB_HOURS - TIME_SAVE_BUFFER_MINS before the limit,
# write a checkpoint, and let the next SLURM array job resume.
#
CHECKPOINT_EVERY       = 1       # save checkpoint every N epochs (1 = safest)
MAX_JOB_HOURS          = 23.5    # leave 30 min for checkpoint write + job overhead
TIME_SAVE_BUFFER_MINS  = 15      # minutes reserved for the final checkpoint I/O
 
# Maximum wall-clock seconds this process is allowed to run.
# MySRForGAN.train() will exit early and save a checkpoint when this is approached.
MAX_RUNTIME_SECONDS = MAX_JOB_HOURS * 3600
 
# --- Optional hyperparameter tuning ---
RUN_TUNING = False
N_TRIALS   = 30
N_SPLITS   = 5
CV_EPOCHS  = 100
 
 
# ============================================================
# Binary file helpers
# ============================================================
 
RECORD_LEN = CONDITION_STEPS + 1        # CONDITION_STEPS values + 1 target value
DTYPE      = np.float32
 
 
def _init_binary_file(bin_path: str, header_path: str, record_len: int) -> None:
    """Create (or truncate) the binary data file and write a fresh JSON header."""
    header = {"record_len": record_len, "n_records": 0}
    with open(header_path, "w") as f:
        json.dump(header, f)
    open(bin_path, "wb").close()
 
 
def _append_chunk_to_binary(
    bin_path: str, header_path: str, trajectories: np.ndarray
) -> None:
    """
    Append a chunk of trajectories (J, N+1) to the flat binary file and
    increment the record count in the header.
    """
    data = trajectories.astype(DTYPE)          # simulator returns float64
    with open(bin_path, "ab") as f:
        f.write(data.tobytes())
    with open(header_path, "r") as f:
        header = json.load(f)
    header["n_records"] += len(trajectories)
    with open(header_path, "w") as f:
        json.dump(header, f)
 
 
# ============================================================
# Checkpoint auto-detection
# ============================================================
 
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Scan *checkpoint_dir* for files matching  checkpoint_epoch_*.pt
    and return the one with the highest epoch number, or None if the
    directory is empty / does not exist.
 
    The epoch encoded in the filename is the *next* epoch to run, so
    passing the returned path directly to  resume_from  is correct.
 
    Example
    -------
    checkpoint_epoch_00001.pt  ← completed epoch 0, will resume at epoch 1
    checkpoint_epoch_00050.pt  ← completed epochs 0-49, resumes at 50
    """
    if not os.path.isdir(checkpoint_dir):
        return None
 
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    candidates = sorted(glob.glob(pattern))   # lexicographic = numeric for zero-padded names
    if not candidates:
        return None
 
    latest = candidates[-1]
    return latest
 
 
def _describe_checkpoint(path: str) -> str:
    """Return a human-readable one-liner about a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location="cpu")
        return (
            f"epoch={ckpt.get('next_epoch', '?')}  "
            f"best_val_sr={ckpt.get('best_val_loss', float('nan')):.6f}  "
            f"patience={ckpt.get('patience_counter', '?')}"
        )
    except Exception as exc:
        return f"(could not read: {exc})"
 
 
# ============================================================
# Streaming Dataset
# ============================================================
 
class HestonBinaryDataset(Dataset):
    """
    Memory-mapped dataset for flat float32 trajectory files.
 
    Data pattern (SRForGAN)
    -----------------------
    Each stored record is a full trajectory [X_0, ..., X_N] of length N+1.
    __getitem__ splits it into:
 
        condition : (CONDITION_STEPS,)  X_0 ... X_{CONDITION_STEPS}  (look-back window)
        target    : (1,)        X_{N_TOTAL}               (h-step-ahead log-price)
 
    returned as  (target, condition) = (y, c)  as expected by
    MySRForGAN.train().
 
    Lazy memmap -- safe for multiprocessing
    ---------------------------------------
    np.memmap is opened on first __getitem__ call, not in __init__.
    This makes the dataset safely picklable by DataLoader worker processes:
    numpy memmaps cannot cross a fork boundary when pre-opened.  Each worker
    opens its own independent file handle after the fork.
    __getstate__ / __setstate__ enforce this contract explicitly.
 
    pin_memory interaction
    ----------------------
    Tensors returned here are regular (pageable) CPU tensors.  The DataLoader
    with pin_memory=True copies them asynchronously into page-locked memory,
    enabling non-blocking PCIe DMA in mySRForGAN.train() via non_blocking=True.
    """
 
    def __init__(self, bin_filepath: str, header_filepath: str):
        # Validate that both files exist before any worker is forked.
        # Catching this here produces a much clearer error than letting
        # the memmap silently fail inside a worker process.
        if not os.path.isfile(header_filepath):
            raise FileNotFoundError(
                f"Header file not found: {header_filepath}\n"
                f"  BASE_DIR = {BASE_DIR}\n"
                "  Ensure $SCRATCH is correctly set and matches the job that "
                "generated the data."
            )
        if not os.path.isfile(bin_filepath):
            raise FileNotFoundError(
                f"Binary data file not found: {bin_filepath}\n"
                f"  BASE_DIR = {BASE_DIR}\n"
                "  Ensure $SCRATCH is correctly set and matches the job that "
                "generated the data."
            )
 
        with open(header_filepath) as f:
            header = json.load(f)
        self._record_len = int(header["record_len"])
        self._n_records  = int(header["n_records"])
        self._filepath   = bin_filepath
        self._mmap       = None                        # opened lazily per worker
 
    def _open_mmap(self) -> None:
        self._mmap = np.memmap(
            self._filepath, dtype=DTYPE, mode="r",
            shape=(self._n_records, self._record_len),
        )
 
    def __len__(self) -> int:
        return self._n_records
 
    def __getitem__(self, idx: int):
        if self._mmap is None:
            self._open_mmap()
        row = self._mmap[idx]                           # (N+1,) float32
        condition = torch.from_numpy(row[:-1].copy())   # (N_STEPS,)
        target    = torch.from_numpy(row[-1:].copy())   # (1,)
        return target, condition                        # (y, c) -- MySRForGAN format
 
    def __getstate__(self):
        """Drop the mmap before pickling so worker processes can re-open it."""
        state = self.__dict__.copy()
        state["_mmap"] = None
        return state
 
    def __setstate__(self, state):
        self.__dict__.update(state)
 
 
# ============================================================
# Step 1: Generate data in chunks
# ============================================================
 
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
 
print(f"Resolved paths:")
print(f"  DATA_DIR       : {DATA_DIR}")
print(f"  MODEL_DIR      : {MODEL_DIR}")
print(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
print(f"  BIN_FILE       : {BIN_FILE}")
print()
 
# Skip generation if a complete binary file already exists.
# This lets you resume a crashed run or reuse a previously generated dataset
# WITHOUT re-generating 10 M paths.
_regen = True
if os.path.exists(BIN_FILE) and os.path.exists(HEADER_FILE):
    try:
        with open(HEADER_FILE) as _f:
            _hdr = json.load(_f)
        if int(_hdr.get("n_records", 0)) == TOTAL_ROWS:
            print(
                f"Binary file already contains {TOTAL_ROWS:,} records -- "
                "skipping data generation.\n"
                f"  Delete '{BIN_FILE}' to force regeneration.\n"
            )
            _regen = False
        else:
            _existing = int(_hdr.get("n_records", 0))
            print(
                f"Binary file is incomplete ({_existing:,} / {TOTAL_ROWS:,} records). "
                "Regenerating from scratch.\n"
            )
    except (json.JSONDecodeError, KeyError):
        print("Header file is corrupted — regenerating data.\n")
 
if _regen:
    print(f"{'='*60}")
    print(f"  Generating {TOTAL_ROWS:,} Heston paths")
    print(f"  {N_CHUNKS} chunks x {CHUNK_SIZE:,} paths each")
    print(f"  N_TOTAL={N_TOTAL} steps, T={T} yr  (dt = T/N_TOTAL = 1/252 per step)")
    print(f"  Stored per record: {RECORD_LEN} floats  "
          f"(X_0…X_{{{CONDITION_STEPS}}} + X_{{{N_TOTAL}}})")
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
            N             = N_TOTAL,
            n_simulations = CHUNK_SIZE,
            seed          = 42 + i,       # reproducible, independent seed per chunk
            scheme        = "milstein",
        )
        # get_paths() returns full log-price trajectories: (CHUNK_SIZE, N_TOTAL+1)
        trajectories = sim.get_paths()
        # Slice: keep the look-back window and the target only.
        # Intermediate steps X_{CONDITION_STEPS} … X_{N_TOTAL-1} are discarded.
        records = np.concatenate([
            trajectories[:, :CONDITION_STEPS],       # (CHUNK_SIZE, CONDITION_STEPS)
            trajectories[:, N_TOTAL : N_TOTAL + 1],  # (CHUNK_SIZE, 1)  ← target
        ], axis=1)                                   # (CHUNK_SIZE, CONDITION_STEPS+1)

        _append_chunk_to_binary(BIN_FILE, HEADER_FILE, records)
 
        total_so_far = (i + 1) * CHUNK_SIZE
        pct = 100.0 * (i + 1) / N_CHUNKS
        print(f"  Chunk {i+1:2d}/{N_CHUNKS}  ({pct:5.1f}%)  --  {total_so_far:,} rows")
 
    print(f"\nData generation complete.  File: {BIN_FILE}\n")
 
 
# ============================================================
# Step 2: Build streaming dataset
# ============================================================
 
full_dataset = HestonBinaryDataset(BIN_FILE, HEADER_FILE)
 
with open(HEADER_FILE) as _f:
    _hdr = json.load(_f)
assert int(_hdr["n_records"]) == TOTAL_ROWS, (
    f"Expected {TOTAL_ROWS:,} records; found {_hdr['n_records']:,}. "
    f"Delete '{BIN_FILE}' and re-run."
)
 
print(f"HestonBinaryDataset ready:")
print(f"  Total records  : {len(full_dataset):,}")
print(f"  Record length  : {full_dataset._record_len}  (= CONDITION_STEPS + 1 = {CONDITION_STEPS + 1})")
print(f"  Condition size : {full_dataset._record_len - 1}  (= Look-back window = {CONDITION_STEPS})")
print(f"  Target size    : 1  (h-step ahead log-price X_N+h)\n")
 
n_val   = int(len(full_dataset) * VAL_FRACTION)
n_train = len(full_dataset) - n_val
train_dataset, val_dataset = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(0),
)
print(f"  Train split    : {n_train:,}  ({100 * (1 - VAL_FRACTION):.0f} %)")
print(f"  Val split      : {n_val:,}   ({100 * VAL_FRACTION:.0f} %)\n")
 
 
# ============================================================
# Step 3 (Optional): Hyperparameter tuning
# ============================================================
 
if RUN_TUNING:
    print(f"{'='*60}")
    print(f"  Hyperparameter tuning  ({N_TRIALS} trials x {N_SPLITS}-fold CV)")
    print(f"{'='*60}\n")
 
    try:
        import optuna
        from sklearn.model_selection import KFold
    except ImportError:
        warnings.warn(
            "optuna or scikit-learn not installed -- skipping tuning. "
            "Install with:  pip install optuna scikit-learn",
            RuntimeWarning,
        )
        RUN_TUNING = False
 
if RUN_TUNING:
 
    def _make_model_for_trial(trial: "optuna.Trial") -> MySRForGAN:
        use_lstm_t = trial.suggest_categorical("use_lstm", [True, False])
        z_dim  = trial.suggest_categorical("z_noise_dim", [16, 32, 64])
        lr     = trial.suggest_float("lr_g", 1e-4, 5e-3, log=True)
        n_samp = trial.suggest_int("n_samples_sr", 5, 20)
        bs     = trial.suggest_categorical("batch_size", [256, 512, 1024])
 
        m = MySRForGAN(
            max_epoch=CV_EPOCHS, batch_size=bs, z_noise_dim=z_dim,
            n_samples_sr=n_samp, scoring_rule=SCORING_RULE, lr_g=lr,
            early_stopping_patience=10,
            name=f"srforgan_trial_{trial.number}",
            use_amp=USE_AMP,
        )
 
        if use_lstm_t:
            h  = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128])
            nl = trial.suggest_int("lstm_n_layers", 1, 3)
            m.set_generator(
                condition_size=CONDITION_STEPS, output_dim=1,
                hidden_dim_rnn=h, n_layers=nl, rnn_layer="lstm",
                dropout=trial.suggest_float("dropout", 0.0, 0.3),
            )
        else:
            hd_str = trial.suggest_categorical(
                "mlp_hidden_dims",
                ["[64,128,64]", "[128,256,128]", "[128,256,256,128,64]"],
            )
            m.set_generator(
                condition_size=CONDITION_STEPS, output_dim=1,
                hidden_dims=json.loads(hd_str),
                use_batch_norm=trial.suggest_categorical("use_bn", [True, False]),
            )
        return m
 
    def _objective(trial: "optuna.Trial") -> float:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold_scores = []
        for _, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_train))):
            fold_train = torch.utils.data.Subset(train_dataset, tr_idx.tolist())
            fold_val   = torch.utils.data.Subset(train_dataset, va_idx.tolist())
            m       = _make_model_for_trial(trial)
            history = m.train(fold_train, val_data=fold_val, save_history=False)
            val_scores = [h["val_sr"] for h in history if h["val_sr"] is not None]
            fold_scores.append(min(val_scores) if val_scores else float("inf"))
        return float(np.mean(fold_scores))
 
    study = optuna.create_study(
        direction="minimize", study_name="srforgan_heston",
        storage="sqlite:///srforgan_study.db", load_if_exists=True,
    )
    study.optimize(_objective, n_trials=N_TRIALS)
 
    print(f"\nBest trial score : {study.best_value:.6f}")
    print(f"Best params      : {study.best_params}\n")
 
    best_p = study.best_params
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
 
 
# ============================================================
# Auto-detect checkpoint to resume from
# ============================================================
 
_resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
 
if _resume_path is not None:
    print(
        f"{'='*60}\n"
        f"  RESUMING from checkpoint:\n"
        f"    {_resume_path}\n"
        f"    {_describe_checkpoint(_resume_path)}\n"
        f"{'='*60}\n"
    )
else:
    print(
        f"{'='*60}\n"
        f"  No checkpoint found in {CHECKPOINT_DIR}\n"
        f"  Starting training from scratch.\n"
        f"{'='*60}\n"
    )
 
 
# ============================================================
# Step 4: Build and train SRForGAN
# ============================================================
 
arch_label = "LSTM  (RnnGenerator)" if USE_LSTM else "MLP   (MyGenerator)"
print(f"{'='*60}")
print(f"  Final training  --  {arch_label}")
print(f"  Dataset        : {n_train:,} train / {n_val:,} val")
print(f"  Max epochs     : {MAX_EPOCH}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  LR generator   : {LR_G}")
print(f"  Scoring rule   : {SCORING_RULE}")
print(f"  m draws / SR   : {N_SAMPLES_SR}")
print(f"  AMP (float16)  : {USE_AMP}")
print(f"  DataLoader workers : {NUM_WORKERS}")
print(f"  Max runtime    : {MAX_JOB_HOURS} h  (buffer: {TIME_SAVE_BUFFER_MINS} min)")
print(f"  Checkpoint dir : {CHECKPOINT_DIR}")
print(f"  Checkpoint every: {CHECKPOINT_EVERY} epoch(s)")
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
    use_amp                 = USE_AMP,
)
 
if USE_LSTM:
    # RnnGenerator: LSTM encodes the condition window (N_STEPS univariate steps),
    # dense head maps [z | h_lstm] to the scalar next log-price.
    model.set_generator(
        condition_size = CONDITION_STEPS,
        output_dim     = 1,
        hidden_dim_rnn = LSTM_HIDDEN_DIM,
        n_layers       = LSTM_N_LAYERS,
        rnn_layer      = "lstm",
        dropout        = LSTM_DROPOUT,
    )
    print(
        f"Generator: RnnGenerator (LSTM)\n"
        f"  condition_size={CONDITION_STEPS}  hidden_dim={LSTM_HIDDEN_DIM}  "
        f"n_layers={LSTM_N_LAYERS}  z_dim={Z_NOISE_DIM}\n"
    )
else:
    # MyGenerator: plain MLP -- condition and noise are concatenated
    model.set_generator(
        condition_size = CONDITION_STEPS,
        output_dim     = 1,
        hidden_dims    = MLP_HIDDEN_DIMS,
        use_batch_norm = True,
    )
    print(
        f"Generator: MyGenerator (MLP)\n"
        f"  condition_size={CONDITION_STEPS}  hidden_dims={MLP_HIDDEN_DIMS}  "
        f"z_dim={Z_NOISE_DIM}\n"
    )
 
total_params = sum(p.numel() for p in model.G.parameters() if p.requires_grad)
print(f"Trainable parameters : {total_params:,}\n")
 
 
# DataLoader configuration
# ========================
# MySRForGAN.train() creates DataLoaders internally without worker settings.
# We temporarily replace torch.utils.data.DataLoader with a wrapper that injects:
#
#   num_workers        -- background workers prefetching data while GPU trains.
#                         Each worker opens its own memmap file handle (safe
#                         because HestonBinaryDataset uses lazy mmap opening).
#
#   pin_memory=True    -- page-locks the host buffer so the PCIe DMA (triggered by
#                         non_blocking=True inside mySRForGAN.train) runs concurrently
#                         with the GPU kernel from the previous batch.
#
#   persistent_workers -- keeps worker processes alive across epochs, avoiding the
#                         fork/join overhead that would otherwise appear every epoch.
#
#   prefetch_factor    -- each worker keeps 2 batches ready ahead of time, hiding
#                         disk and decompression latency entirely.
#
# The original class is restored immediately after training.
 
_OrigDataLoader = torch.utils.data.DataLoader
 
 
def _streaming_dataloader(ds, **kw):
    if NUM_WORKERS > 0:
        kw.setdefault("num_workers",        NUM_WORKERS)
        kw.setdefault("pin_memory",         True)
        kw.setdefault("persistent_workers", True)
        kw.setdefault("prefetch_factor",    2)
    return _OrigDataLoader(ds, **kw)
 
 
torch.utils.data.DataLoader = _streaming_dataloader   # patch
 
history = model.train(
    data         = train_dataset,
    val_data     = val_dataset,
    save_history = True,
    # ── Checkpoint / time-limit arguments ─────────────────────────────────
    checkpoint_dir           = CHECKPOINT_DIR,
    checkpoint_every         = CHECKPOINT_EVERY,
    resume_from              = _resume_path,          # None → start fresh
    max_runtime_seconds      = MAX_RUNTIME_SECONDS,
    time_save_buffer_seconds = TIME_SAVE_BUFFER_MINS * 60,
)
 
torch.utils.data.DataLoader = _OrigDataLoader          # restore
 
 
# ============================================================
# Step 5: Save final model
# ============================================================
#
# This block is reached only when training completes ALL epochs within
# the time limit (or early-stopping fires).  If the time limit fires
# first, train() returns early and writes a checkpoint instead; the
# final model save happens in the *last* job of the chain.
 
_final_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
_completed_all = (
    _final_ckpt is not None
    and (int(os.path.basename(_final_ckpt)
            .replace("checkpoint_epoch_", "")
            .replace(".pt", "")) >= MAX_EPOCH or model.early_stopped))
 
if _completed_all:
    print("\nAll epochs completed — saving final model.")
    model.save_models(MODEL_DIR)
 
    training_cfg = {
        # Simulation
        "X0_range": list(X0_RANGE), "mu_range": list(MU_RANGE),
        "v0_range": list(V0_RANGE), "kappa_range": list(KAPPA_RANGE),
        "theta_range": list(THETA_RANGE), "sigma_v_range": list(SIGMA_V_RANGE),
        "rho_range": list(RHO_RANGE),
        "N_steps": CONDITION_STEPS, "T": T,
        "total_rows": TOTAL_ROWS, "chunk_size": CHUNK_SIZE,
        # Architecture
        "use_lstm": USE_LSTM, "z_noise_dim": Z_NOISE_DIM,
        "lstm_hidden_dim": LSTM_HIDDEN_DIM if USE_LSTM else None,
        "lstm_n_layers":   LSTM_N_LAYERS   if USE_LSTM else None,
        "lstm_dropout":    LSTM_DROPOUT    if USE_LSTM else None,
        "mlp_hidden_dims": MLP_HIDDEN_DIMS if not USE_LSTM else None,
        # Training
        "max_epoch": MAX_EPOCH, "batch_size": BATCH_SIZE,
        "lr_g": LR_G, "n_samples_sr": N_SAMPLES_SR,
        "scoring_rule": SCORING_RULE, "early_stop_pat": EARLY_STOP_PAT,
        "val_fraction": VAL_FRACTION,
        # GPU
        "use_amp": USE_AMP, "num_workers": NUM_WORKERS,
        "device": _device_str,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    train_cfg_path = os.path.join(MODEL_DIR, "srforgan_heston_training_config.json")
    with open(train_cfg_path, "w") as _f:
        json.dump(training_cfg, _f, indent=2)
    print(f"Training config saved to {train_cfg_path}")
    print(f"\nDone.  Models saved to {MODEL_DIR}/srforgan_heston_*\n")
 
else:
    print(
        "\nTime-limit checkpoint saved.  Submit the next SLURM job to continue.\n"
        f"  Latest checkpoint : {_final_ckpt}\n"
        "  The next job will auto-detect and resume from it.\n"
    )
 
 
# ============================================================
# Step 6: Sanity check -- sample from the trained generator
# ============================================================
#
# Only run when training actually finished (not on a mid-run checkpoint exit).
 
if _completed_all:
    print("--- Sanity check: draw samples from trained generator ---")
    device = model.DEVICE
    model.G.eval()
 
    _val_loader = _OrigDataLoader(val_dataset, batch_size=8, shuffle=True)
    y_batch, c_batch = next(iter(_val_loader))
 
    # non_blocking=True is safe even without pin_memory (falls back to blocking)
    y_batch = y_batch.to(device, dtype=torch.float32, non_blocking=True)
    c_batch = c_batch.to(device, dtype=torch.float32, non_blocking=True)
 
    with torch.no_grad():
        m_draws = 200
        c_exp = c_batch.unsqueeze(1).expand(-1, m_draws, -1).reshape(8 * m_draws, -1)
        z     = torch.randn(8 * m_draws, model.z_dim, device=device)
 
        _amp_device = "cuda" if USE_AMP else "cpu"
        with torch.autocast(device_type=_amp_device, dtype=torch.float16, enabled=USE_AMP):
            fake = model.G(c_exp, z).view(8, m_draws, 1)
 
    pred_mean = fake.float().mean(dim=1).cpu().numpy()
    true_vals = y_batch.cpu().numpy()
    mae = float(np.abs(pred_mean - true_vals).mean())
    print(f"  Predictive mean MAE (8 samples, m={m_draws} draws): {mae:.6f}")
    print(
        "  NOTE: MAE ~= 0 is not the training objective.\n"
        "        The goal is correct spread / calibration of the predictive distribution."
    )
 