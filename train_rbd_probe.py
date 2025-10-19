"""Train linear probes on SleepFM embeddings for CAP RBD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

from config_cap_paths import PROCESSED_DIR


def _load_embedding_matrix(paths: Sequence[str]) -> np.ndarray:
    vectors = [np.load(path) for path in paths]
    return np.stack(vectors, axis=0)


def _confidence_interval(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return math.nan, (math.nan, math.nan)
    mean_val = float(np.mean(valid))
    if len(valid) == 1:
        return mean_val, (mean_val, mean_val)
    std = float(np.std(valid, ddof=1))
    half_width = NormalDist().inv_cdf(1 - alpha / 2) * std / math.sqrt(len(valid))
    return mean_val, (mean_val - half_width, mean_val + half_width)


def _prepare_targets(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    stage_codes = frame["stage_code"].astype(str).str.upper()
    rbd_events = frame.get("rbd_event", pd.Series(False, index=frame.index)).astype(bool)
    y_stage = stage_codes.to_numpy()
    y_rbd = ((stage_codes == "REM") & rbd_events.to_numpy()).astype(int)
    return y_stage, y_rbd


def _ensure_valid_splits(groups: Sequence[str], desired_splits: int) -> GroupKFold:
    unique_subjects = np.unique(groups)
    if unique_subjects.size < 2:
        raise ValueError("At least two subjects are required for cross-validation")
    n_splits = min(desired_splits, unique_subjects.size)
    if n_splits < 2:
        n_splits = 2
    return GroupKFold(n_splits=n_splits)


def _train_stage_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    gkf = _ensure_valid_splits(groups, desired_splits=5)

    roc_values: List[float] = []
    pr_values: List[float] = []

    for train_idx, test_idx in gkf.split(X, y_encoded, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        present_classes = np.unique(y_test)
        if present_classes.size < 2:
            continue
        y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
        roc_values.append(roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr"))
        pr_values.append(average_precision_score(y_test_bin, probs, average="macro"))

    return {
        "roc_auc": _confidence_interval(roc_values),
        "auprc": _confidence_interval(pr_values),
    }


def _train_rbd_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    gkf = _ensure_valid_splits(groups, desired_splits=5)
    roc_values: List[float] = []
    pr_values: List[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        roc_values.append(roc_auc_score(y_test, probs))
        pr_values.append(average_precision_score(y_test, probs))

    return {
        "roc_auc": _confidence_interval(roc_values),
        "auprc": _confidence_interval(pr_values),
    }


def train_probes(manifest_path: Path, embeddings_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings index not found at {embeddings_path}. "
            "Run generate_embeddings_bas.py first to create embeddings_bas.csv."
        )
    embeddings = pd.read_csv(embeddings_path)

    merged = manifest.merge(
        embeddings,
        on=["subject_id", "epoch_idx", "stage_code"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No overlapping entries between manifest and embeddings")

    X = _load_embedding_matrix(merged["emb_path"].tolist())
    y_stage, y_rbd = _prepare_targets(merged)
    groups = merged["subject_id"].to_numpy()

    stage_metrics = _train_stage_probe(X, y_stage, groups)
    rbd_metrics = _train_rbd_probe(X, y_rbd, groups)

    print("Stage classification metrics (macro averages)")
    for metric, (mean_val, (lower, upper)) in stage_metrics.items():
        print(f"  {metric}: {mean_val:.4f} (95% CI: {lower:.4f} – {upper:.4f})")

    print("\nRBD detection metrics")
    for metric, (mean_val, (lower, upper)) in rbd_metrics.items():
        print(f"  {metric}: {mean_val:.4f} (95% CI: {lower:.4f} – {upper:.4f})")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "manifest_cap_rbd_bas_only.csv")
    parser.add_argument("--embeddings", type=Path, default=PROCESSED_DIR / "embeddings_bas.csv")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    train_probes(args.manifest, args.embeddings)


if __name__ == "__main__":
    main()

