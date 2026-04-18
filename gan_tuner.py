"""
gan_tuner.py
============
Data validation + Optuna hyperparameter tuning for MyCGAN / MyCWGAN.

Training and evaluation are delegated entirely to the existing
model.train() and model.generate() methods — no logic is duplicated here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import optuna
import torch
from torch.utils.data import Subset, TensorDataset

from GANComponents import _unwrap
from myCGAN import MyCGAN
from myCWGAN import MyCWGAN
from utilities import compute_js, compare_simulated_pdfs, DataSimulator

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data validation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    is_valid: bool = True
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self):
        if not self.is_valid:
            raise ValueError("Data validation failed:\n  " + "\n  ".join(self.errors))

    def summary(self) -> str:
        lines = [f"Validation {'PASSED' if self.is_valid else 'FAILED'}"]
        lines += [f"  ✗ {e}" for e in self.errors]
        lines += [f"  ⚠ {w}" for w in self.warnings]
        return "\n".join(lines)


class DataValidator:
    """
    Inspect a TensorDataset before training.

    Hard errors  → is_valid = False  (NaN, Inf, shape mismatch, too few samples)
    Warnings     → non-fatal         (near-zero variance columns, large value range)
    """

    def __init__(self, low_variance_threshold: float = 1e-6, min_samples: int = 64,
                 tensor_names: Optional[List[str]] = None):
        self.low_variance_threshold = low_variance_threshold
        self.min_samples = min_samples
        self.tensor_names = tensor_names or []

    def validate(self, dataset: TensorDataset) -> ValidationReport:
        report = ValidationReport()
        tensors = dataset.tensors
        n = tensors[0].size(0)

        if n < self.min_samples:
            report.is_valid = False
            report.errors.append(f"Only {n} samples; minimum is {self.min_samples}.")

        for i, t in enumerate(tensors):
            name = self.tensor_names[i] if i < len(self.tensor_names) else f"tensor_{i}"
            arr = t.detach().cpu().float().numpy()

            if np.isnan(arr).any():
                report.is_valid = False
                report.errors.append(f"{name}: contains NaN.")
            if np.isinf(arr).any():
                report.is_valid = False
                report.errors.append(f"{name}: contains Inf.")
            if t.size(0) != n:
                report.is_valid = False
                report.errors.append(f"{name}: row count {t.size(0)} != {n}.")

            safe = np.nan_to_num(arr)
            low_var = (safe.std(axis=0) < self.low_variance_threshold).sum()
            if low_var:
                report.warnings.append(f"{name}: {low_var} near-zero-variance column(s).")
            if np.abs(safe).max() > 1e4:
                report.warnings.append(f"{name}: max |value| > 1e4 — consider normalising.")

        return report


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_tensordataset(subset: Subset) -> TensorDataset:
    """Re-materialise a Subset as a plain TensorDataset."""
    idx = subset.indices
    return TensorDataset(*(t[idx] for t in subset.dataset.tensors))


def _score(model: MyCGAN, val_ds: TensorDataset,
           true_pdfs: np.ndarray, bins: np.ndarray|None = None) -> float:
    """
    Score a trained fold using the same protocol as the test pipeline:
      1. model.generate(val_ds, get_pdf=True) → (n, 1000) raw samples.
      2. Each row is histogrammed into n_bins using fold-derived bin edges.
      3. Per-condition JS divergence against the pre-computed analytic PDFs.

    Parameters
    ----------
    true_pdfs : (n, n_bins) analytic PDFs for each validation condition,
                pre-computed once in __init__ via sim.get_pdf().
    bins    : bin edges
    """
    real_pdfs = true_pdfs
    _, gen_pdfs = model.generate(val_ds, get_pdf=True, bins = bins)   # (n, 1000) raw samples

    if bins is None:
        real_pdfs, gen_pdfs, bin_edges_list = compare_simulated_pdfs(true_pdfs, gen_pdfs)

    js = compute_js(gen_pdfs, real_pdfs, is_log=False)
    return float(np.nanmean(js))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter tuner
# ──────────────────────────────────────────────────────────────────────────────

class GANHyperparameterTuner:
    """
    Optuna-driven k-fold hyperparameter search for MyCGAN / MyCWGAN.

    Each trial:
      1. Samples a hyperparameter configuration.
      2. Runs k-fold CV by calling model.train() / model.generate() directly.
      3. Returns the mean validation score across folds.

    Parameters
    ----------
    dataset        : Full TensorDataset (target, condition).
    model_type     : 'cgan' or 'cwgan'.
    input_size     : Feature count of the target tensor.
    condition_size : Feature count of the condition tensor.
    output_dim     : Generator output size.
    n_trials       : Optuna trials.
    n_splits       : CV folds.
    cv_epochs      : Epochs per fold (keep low for speed; tune separately).
    study_name     : Optuna study name.
    storage        : Optional persistent storage URL (e.g. 'sqlite:///study.db').
    """

    def __init__(
        self,
        dataset: TensorDataset,
        sim: DataSimulator,                                      
        model_type: Literal["cgan", "cwgan"] = "cwgan",
        input_size: int = 260,
        condition_size: int = 22,
        output_dim: int = 2,
        n_trials: int = 50,
        n_splits: int = 5,
        cv_epochs: int = 100,
        n_bins: int|None = None,
        study_name: str = "gan_tuning",
        storage: Optional[str] = None,
    ):
        if model_type not in ("cgan", "cwgan"):
            raise ValueError(f"model_type must be 'cgan' or 'cwgan', got '{model_type}'.")

        self.dataset        = dataset
        self.model_type     = model_type
        self.input_size     = input_size
        self.condition_size = condition_size
        self.output_dim     = output_dim
        self.n_trials       = n_trials
        self.n_splits       = n_splits
        self.cv_epochs      = cv_epochs
        self.study_name     = study_name
        self.storage        = storage
        self.sim            = sim
        self.n_bins         = n_bins

        # Pre-compute shuffled fold indices once so all trials see the same splits.
        n = len(dataset)
        idx = np.random.default_rng(42).permutation(n)
        fold_size = n // n_splits
        self._folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_splits):
            val_start = k * fold_size
            val_end   = val_start + fold_size if k < n_splits - 1 else n
            val_idx   = idx[val_start:val_end]
            train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
            self._folds.append((train_idx, val_idx))

        # Pre-compute analytic PDFs for every fold's validation set once.
        # These only depend on the paths (conditions), not the model, so they
        # are identical across all Optuna trials — computed once, reused per trial.
        all_paths = dataset.tensors[1].cpu().numpy()          # (J, N-1)
        self._fold_pdfs: List[np.ndarray] = []
        for _, val_idx in self._folds:
            val_paths = all_paths[val_idx]                    # (n_val, N-1)
            true_pdfs = sim.get_pdf(
                n_bins=n_bins,
                P=val_paths[:, -1],                           # last observed value per path
                n_steps_ahead=1,
            )                                                 # (n_val, n_bins)
            self._fold_pdfs.append(true_pdfs)

    # ── search space ──────────────────────────────────────────────────────────

    def _suggest(self, trial: optuna.Trial) -> Dict:
        n_layers   = trial.suggest_int("n_layers", 1, 4)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])

        params = dict(
            z_noise_dim = trial.suggest_categorical("z_noise_dim", [24, 48, 96, 192]),
            batch_size  = trial.suggest_categorical("batch_size",  [16, 32, 64, 128]),
            n_critic    = trial.suggest_int("n_critic", 1, 3 if self.model_type == "cgan" else 10),
            lr_g        = trial.suggest_float("lr_g", 1e-5, 1e-2, log=True),
            lr_d        = trial.suggest_float("lr_d", 1e-5, 1e-2, log=True),
            # shared architecture axes for G and D
            hidden_dims = [hidden_dim] * n_layers,
            activation  = trial.suggest_categorical("activation", ["leaky_relu", "relu"]),
            dropout     = trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
        )
        if self.model_type == "cwgan":
            params["lambda_gp"] = trial.suggest_categorical("lambda_gp", [1.0, 10.0, 40.0])
        return params

    # ── model factory ─────────────────────────────────────────────────────────

    def _build(self, params: Dict) -> MyCGAN:
        """Instantiate and configure a fresh model from a parameter dict."""
        common = dict(
            max_epoch   = self.cv_epochs,
            batch_size  = params["batch_size"],
            n_critic    = params["n_critic"],
            z_noise_dim = params["z_noise_dim"],
            lr_g        = params["lr_g"],
            lr_d        = params["lr_d"],
        )
        model = (MyCWGAN(**common, lambda_gp=params.get("lambda_gp", 10.0))
                 if self.model_type == "cwgan" else MyCGAN(**common))

        arch = dict(hidden_dims=params["hidden_dims"],
                    activation=params["activation"],
                    dropout=params["dropout"])

        model.set_generator(condition_size=self.condition_size,
                            output_dim=self.output_dim, **arch)
        model.set_discriminator(input_size=self.input_size,
                                condition_size=self.condition_size,
                                output_dim=1, **arch)
        return model

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _objective(self, trial: optuna.Trial) -> float:
        params = self._suggest(trial)
        fold_scores = []

        for k, (train_idx, val_idx) in enumerate(self._folds):
            train_ds = _to_tensordataset(Subset(self.dataset, train_idx.tolist()))
            val_ds   = _to_tensordataset(Subset(self.dataset, val_idx.tolist()))

            model = self._build(params)
            try:
                model.train(train_ds)                   
                score = _score(model, val_ds, self._fold_pdfs[k], self.n_bins)
            except Exception as exc:
                warnings.warn(f"Trial {trial.number} fold {k} failed: {exc}")
                score = float("inf")

            fold_scores.append(score)
            trial.report(float(np.nanmean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean = float(np.nanmean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        logger.info(f"Trial {trial.number} | mean={mean:.5f} | folds={[round(s,5) for s in fold_scores]}")
        return mean

    # ── public interface ──────────────────────────────────────────────────────

    def tune(self) -> optuna.Study:
        """Run the Optuna study and return it."""
        study = optuna.create_study(
            direction  = "minimize",
            sampler    = optuna.samplers.TPESampler(multivariate=True, seed=42),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            study_name = self.study_name,
            storage    = self.storage,
            load_if_exists=True,
        )
        study.optimize(self._objective, n_trials=self.n_trials,
                       show_progress_bar=True, catch=(RuntimeError, ValueError))
        logger.info(self._summary(study))
        return study

    def build_best_model(self, study: optuna.Study, max_epoch: int = 200) -> MyCGAN:
        """Reconstruct the best architecture for final full-dataset training."""
        p = study.best_trial.params
        params = dict(
            z_noise_dim = p["z_noise_dim"],
            batch_size  = p["batch_size"],
            n_critic    = p["n_critic"],
            lr_g        = p["lr_g"],
            lr_d        = p["lr_d"],
            hidden_dims = [p["hidden_dim"]] * p["n_layers"],
            activation  = p["activation"],
            dropout     = p["dropout"],
        )
        if "lambda_gp" in p:
            params["lambda_gp"] = p["lambda_gp"]

        model = self._build(params)
        model.max_epoch = max_epoch
        return model

    def _summary(self, study: optuna.Study) -> str:
        b = study.best_trial
        lines = [
            f"\n{'='*55}",
            f"Best trial #{b.number}  score={b.value:.6f}",
            f"Fold scores: {b.user_attrs.get('fold_scores')}",
            "Hyperparameters:",
        ] + [f"  {k} = {v}" for k, v in b.params.items()] + ["="*55]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Convenience one-liner
# ──────────────────────────────────────────────────────────────────────────────

def validate_and_tune(
    dataset: TensorDataset,
    sim,                                          # DataSimulator instance
    model_type: Literal["cgan", "cwgan"],
    input_size: int,
    condition_size: int,
    output_dim: int,
    n_trials: int = 50,
    n_splits: int = 5,
    cv_epochs: int = 30,
    n_bins: int = 100,
    max_epoch_final: int = 200,
    study_name: str = "gan_tuning",
    storage: Optional[str] = None,
) -> Tuple[MyCGAN, optuna.Study, ValidationReport]:
    """
    Validate data, tune hyperparameters, return best model ready for final training.

    Example
    -------
    >>> best, study, report = validate_and_tune(
    ...     dataset, sim, "cwgan",
    ...     input_size=1, condition_size=21, output_dim=1,
    ...     n_trials=40, cv_epochs=20, storage="sqlite:///study.db",
    ... )
    >>> print(report.summary())
    >>> best.train(dataset)
    >>> best.save_models("./models/best")
    """
    report = DataValidator(tensor_names=["target", "condition"]).validate(dataset)
    print(report.summary())
    report.raise_if_invalid()

    tuner = GANHyperparameterTuner(
        dataset=dataset, sim=sim, model_type=model_type,
        input_size=input_size, condition_size=condition_size, output_dim=output_dim,
        n_trials=n_trials, n_splits=n_splits, cv_epochs=cv_epochs,
        n_bins=n_bins, study_name=study_name, storage=storage,
    )
    study      = tuner.tune()
    best_model = tuner.build_best_model(study, max_epoch=max_epoch_final)
    return best_model, study, report


# ──────────────────────────────────────────────────────────────────────────────
# 5. Example
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utilities import DataSimulator, prepare_data

    N_BINS = 100
    sim = DataSimulator(
        X0_range=(0.0, 0.0), mu_range=(0.0, 0.0),
        sigma_range=(0.01, 0.8), T=round(22/252, 3),
        N=22, n_simulations=500000, seed=42,
    )
    trajectories = sim.get_paths()
    paths   = trajectories[:, :-1]   # (J, N-1) — condition
    targets = trajectories[:, -1:]   # (J,  1 ) — target
    dataset, _, _ = prepare_data(targets, paths)

    best, study, report = validate_and_tune(
        dataset, sim, "cwgan",
        input_size=targets.shape[1],
        condition_size=paths.shape[1],
        output_dim=targets.shape[1],
        n_bins=N_BINS,
        n_trials=30, n_splits=3, cv_epochs=20, max_epoch_final=150,
        storage="sqlite:///cwgan_study.db",
    )
    best.train(dataset)
    best.save_models("./models/best_cwgan")