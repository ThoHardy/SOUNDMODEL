from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np
from sklearn.model_selection import KFold


def _prepare_import_path(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


DEFAULT_CHECKPOINT_NAME = "5CV_Active_late.json"
DEFAULT_TASK = "Active"
DEFAULT_SUB_IDS = ["01", "02", "03", "05", "06", "07", "08", "09", "11", "12", "13", "14", "15", "17", "19", "20", "22", "23", "24", "25"]


def _reorganize_training_state(state_train: dict[int, np.ndarray], n_categories: int) -> None:
    no_stim_sliced = np.concatenate([state_train[0][:, i * 50 : (i + 1) * 50] for i in range(5)])
    pre_stim_sliced = np.concatenate([state_train[cat][:, :50] for cat in range(1, n_categories)])
    state_train[0] = np.concatenate((no_stim_sliced, pre_stim_sliced))


def _build_constant_stim_test_sets(
    state_test: dict[int, np.ndarray], input_test: dict[int, np.ndarray], n_categories: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    state_test_constant_stim = {cat: state_test[cat][:, 75:100] for cat in range(1, n_categories)}
    state_test_constant_stim[0] = np.concatenate(
        (
            np.concatenate([state_test[0][:, i * 50 : (i + 1) * 50] for i in range(5)]),
            np.concatenate([state_test[cat][:, :50] for cat in range(1, n_categories)]),
        )
    )

    input_test_constant_stim = {cat: input_test[cat][:, 75:100] for cat in range(1, n_categories)}
    input_test_constant_stim[0] = np.zeros_like(state_test_constant_stim[0])
    return state_test_constant_stim, input_test_constant_stim


def crossval_null_loglik(state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]):
    """Compute 5-fold cross-validated log-likelihoods for the null model only."""

    from LEAD import fitting_tools

    n_categories = len(list(state_series.keys()))
    categories = list(range(n_categories))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    trial_indices = np.arange(np.min([state_series[cat].shape[0] for cat in categories]))

    test_lls_null = []
    cv_params = []

    for train_idx, test_idx in kf.split(trial_indices):
        state_train, input_train = {}, {}
        state_test, input_test = {}, {}

        for cat in categories:
            state_train[cat] = state_series[cat][train_idx]
            input_train[cat] = input_series[cat][train_idx]
            state_test[cat] = state_series[cat][test_idx]
            input_test[cat] = input_series[cat][test_idx]

        _reorganize_training_state(state_train, n_categories)
        input_train[0] = np.zeros_like(state_train[0])

        null = fitting_tools.clever_fit_null(
            state_train=state_train,
            input_train=input_train,
            input_start_index=75,
            input_stop_index=100,
        )

        state_test_constant_stim, input_test_constant_stim = _build_constant_stim_test_sets(
            state_test, input_test, n_categories
        )

        ll_null = null.loglikelihood(state_test_constant_stim, input_test_constant_stim)
        test_lls_null.append(ll_null)
        cv_params.append(null.get_params())

    return np.array(test_lls_null), cv_params


def _load_checkpoint(checkpoint_path: Path) -> dict:
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)

    if "detailed" not in checkpoint:
        raise KeyError(f"Missing 'detailed' in checkpoint: {checkpoint_path}")
    if "results_cv" not in checkpoint:
        raise KeyError(f"Missing 'results_cv' in checkpoint: {checkpoint_path}")

    return checkpoint


def _write_checkpoint_atomic(checkpoint_path: Path, checkpoint: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=checkpoint_path.parent, delete=False) as tmp_file:
        json.dump(checkpoint, tmp_file, indent=2)
        tmp_file.write("\n")
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(checkpoint_path)


def backfill_checkpoint(
    checkpoint_path: str | Path | None = None,
    *,
    task: str = DEFAULT_TASK,
    sub_ids: Iterable[str] = DEFAULT_SUB_IDS,
    repo_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict:
    """Backfill null-model CV results into an existing 5-fold checkpoint JSON."""

    if checkpoint_path is None:
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]
        checkpoint_path = Path(repo_root) / DEFAULT_CHECKPOINT_NAME
    else:
        checkpoint_path = Path(checkpoint_path)

    if repo_root is None:
        repo_root = checkpoint_path.parent
    repo_root = Path(repo_root)

    if data_root is None:
        data_root = repo_root
    data_root = Path(data_root)

    _prepare_import_path(repo_root)
    import LEAD as lead  # noqa: F401  # Imported for side effects and consistency with notebook context.

    checkpoint = _load_checkpoint(checkpoint_path)
    detailed = checkpoint["detailed"]

    sub_ids = list(sub_ids)
    if len(detailed) > len(sub_ids):
        raise ValueError(
            f"Checkpoint contains {len(detailed)} detailed entries but only {len(sub_ids)} subject IDs were provided."
        )

    for part, entry in enumerate(detailed):
        if entry.get("part") != part:
            raise ValueError(f"Detailed entry {part} has mismatched part index: {entry.get('part')}")

        data_ref = f"myEpochs_{task}/Epoch_{sub_ids[part]}-epo.fif"
        epochs_file = data_root / data_ref

        if not epochs_file.exists():
            raise FileNotFoundError(
                f"Missing epochs file for participant {part}: {epochs_file}. "
                "Pass --data-root or data_root when the EEG epochs live outside the repository root."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            state_series = lead.STG(epochs_file, tmin=300, tmax=500)
            n_categories = len(list(state_series.keys()))
            categories = list(range(n_categories))

        one_input = np.concatenate((np.zeros(75), np.ones(25), np.zeros(150)))
        input_series = {
            cat: np.stack([one_input for _ in range(state_series[cat].shape[0])]) for cat in categories
        }

        lls_null, _ = crossval_null_loglik(state_series, input_series)
        entry["ll_null_folds"] = lls_null.tolist()
        entry["ll_null_mean"] = float(np.mean(lls_null))

    _write_checkpoint_atomic(checkpoint_path, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill null-model CV log-likelihoods into an existing checkpoint JSON.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to the checkpoint JSON to update.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name used to build the epoch path.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repository root containing myEpochs_<task>/.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing myEpochs_<task>/.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    repo_root = args.repo_root
    if checkpoint_path is None and repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    backfill_checkpoint(checkpoint_path=checkpoint_path, task=args.task, repo_root=repo_root, data_root=args.data_root)


if __name__ == "__main__":
    main()
