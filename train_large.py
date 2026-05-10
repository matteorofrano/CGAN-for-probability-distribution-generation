"""
train_large.py
==============
End-to-end ForGAN training pipeline for datasets that exceed available RAM
(e.g. 1 million GBM paths).
 
Workflow
--------
1. GENERATE   – Simulate data in chunks; append each chunk to one .bin file.
                 The first chunk fixes the shared bin edges (saved to JSON).
                 Every subsequent chunk loads those edges before calling get_pdf,
                 so all records use an identical discretisation.
 
2. STREAM     – Build a BinaryDataset that memory-maps the .bin file via O(1)
                 seeks.  DataLoader workers read individual records on demand;
                 the full dataset never sits in RAM simultaneously.
 
3. TUNE       – Run BinaryGANHyperparameterTuner (Optuna + k-fold CV).
                 Training folds stream from disk; validation folds are loaded
                 into RAM fold-by-fold (~9 MB each for the defaults below).
                 Ground-truth PDFs for fold scoring are read straight from the
                 binary file — no sim.mu / sim.sigma shape-mismatch.
 
4. TRAIN BEST – Retrain the best architecture on the full 1 M-row dataset.
 
Usage (HPC)
-----------
    python train_large.py
 
    # Optionally set OMP_NUM_THREADS / CUDA_VISIBLE_DEVICES before calling.
    # Increase NUM_WORKERS to exploit fast NVMe storage (e.g. 4–8).
"""
from __future__ import annotations
 
import json
import os
 
import torch.utils.data
import numpy as np
import torch
 
from utilities import DataSimulator, BinaryDataset
from gan_tuner import binary_validate_and_tune
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
# GBM simulation parameters
X0_RANGE    = (0.0, 0.0)
MU_RANGE    = (0.0, 0.0)
SIGMA_RANGE = (0.05, 1.0)
N_STEPS     = 22                         # number of time steps
T           = round(22 / 252, 3)         # ~1 trading month
 
# Data volume
TOTAL_ROWS  = 1000000
CHUNK_SIZE  =   100000                  # rows per simulation batch
N_CHUNKS    = TOTAL_ROWS // CHUNK_SIZE   # 10 chunks of 100 K each
 
# Discretisation
N_BINS      = 100                        # histogram bins for PDF comparison
 
# File paths
DATA_DIR    = "data"
BIN_FILE    = os.path.join(DATA_DIR, "training_1M")   # .bin appended automatically
CONFIG_FILE = os.path.join(DATA_DIR, "training_bins.json")
 
# Training
N_TRIALS         = 50
N_SPLITS         = 5
CV_EPOCHS        = 60
MAX_EPOCH_FINAL  = 100
NUM_WORKERS      = 4     # DataLoader workers; 0 = main thread (safe on all OSes)
                         # Increase to 4–8 on HPC nodes with NVMe storage.
STUDY_STORAGE    = "sqlite:///cwgan_study.db"
 
# ── Step 1: Generate data in chunks ───────────────────────────────────────────
 
os.makedirs(DATA_DIR, exist_ok=True)
 
print(f"{'='*60}")
print(f"  Generating {TOTAL_ROWS:,} rows in {N_CHUNKS} chunks of {CHUNK_SIZE:,}")
print(f"{'='*60}\n")
 
for i in range(N_CHUNKS):
    sim = DataSimulator(
        X0_range=X0_RANGE,
        mu_range=MU_RANGE,
        sigma_range=SIGMA_RANGE,
        T=T, N=N_STEPS,
        n_simulations=CHUNK_SIZE,
        seed=42 + i,          # different seed per chunk → independent paths
    )
    sim.get_paths()
    
 
    if i == 0:
        # ── First chunk: establish the global bin edges ────────────────────
        # Bins are computed from [global_mean - 4σ, global_mean + 4σ] over
        # this chunk.  Using 100 K paths is sufficient to cover the tails of
        # the full sigma range; the JSON is reused by all subsequent chunks.
        sim.get_pdf(n_steps_ahead=1, n_bins=N_BINS)
        sim.save_configuration(CONFIG_FILE)
        print(f"  [Chunk 0] Bins ({N_BINS} bins) saved to {CONFIG_FILE}")
    else:
        # ── Subsequent chunks: load the fixed bins before calling get_pdf ──
        # DataSimulator.get_pdf() short-circuits to the loaded bins when
        # sim.bins is not None, so all records share the same edges.
        sim.load_configuration(CONFIG_FILE)
        sim.get_pdf(n_steps_ahead=1)   # n_bins is ignored; loaded bins are used
 
    sim.save_binary_file(BIN_FILE)
    total_so_far = (i + 1) * CHUNK_SIZE
    print(f"  Chunk {i + 1:2d}/{N_CHUNKS} appended — {total_so_far:,} rows total")
 
print(f"\nData generation complete.  File: {BIN_FILE}.bin\n")
 
# ── Step 2: Build streaming dataset ───────────────────────────────────────────
 
dataset = BinaryDataset(BIN_FILE)
print(f"BinaryDataset ready:")
print(f"  Total records  : {len(dataset):,}")
print(f"  Path length    : {dataset._len_path}  (= N + 1 = {N_STEPS + 1})")
print(f"  PDF  bins      : {dataset._len_pdf}   (= N_BINS = {N_BINS})")
 
# Load shared bin edges (needed by the tuner for JS scoring)
with open(CONFIG_FILE) as f:
    bins = np.array(json.load(f)["bins"], dtype=np.float32)  # shape (N_BINS + 1,)
 
print(f"  Bin edges      : [{bins[0]:.4f}, …, {bins[-1]:.4f}]  ({len(bins)} edges)\n")
 
# ── Step 3: Hyperparameter search with k-fold CV ───────────────────────────────
 
print(f"{'='*60}")
print(f"  Hyperparameter tuning  ({N_TRIALS} trials × {N_SPLITS}-fold CV)")
print(f"{'='*60}\n")
 
best, study, report = binary_validate_and_tune(
    dataset        = dataset,
    bins           = bins,
    model_type     = "cwgan",
    condition_size = N_STEPS,        # path[:-1] has N_STEPS = 22 values
    n_trials       = N_TRIALS,
    n_splits       = N_SPLITS,
    cv_epochs      = CV_EPOCHS,
    num_workers    = NUM_WORKERS,
    max_epoch_final= MAX_EPOCH_FINAL,
    study_name     = "cwgan_1M",
    storage        = STUDY_STORAGE,
)
 
print("\n" + report.summary())
print(f"\nBest trial score : {study.best_value:.6f}")
print(f"Best params      : {study.best_params}\n")
 
# ── Step 4: Final training on the full 1 M-row dataset ───────────────────────
 
print(f"{'='*60}")
print(f"  Final training on {len(dataset):,} rows")
print(f"{'='*60}\n")
 
# best.train() now accepts a BinaryDataset directly (isinstance check relaxed).
# DataLoader will stream records from disk for every epoch.
# After binary_validate_and_tune, retrain with workers
_orig_dl = torch.utils.data.DataLoader
def _patched_dl(ds, **kw):
    kw.setdefault('num_workers', NUM_WORKERS)
    kw.setdefault('pin_memory', NUM_WORKERS > 0)
    return _orig_dl(ds, **kw)
torch.utils.data.DataLoader = _patched_dl
best.train(dataset)
torch.utils.data.DataLoader = _orig_dl
 
os.makedirs("models", exist_ok=True)
best.save_models("./models/best_cwgan")
print("\nDone.  Models saved to ./models/best_cwgan")