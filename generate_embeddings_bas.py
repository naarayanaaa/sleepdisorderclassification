"""Generate SleepFM BAS embeddings for the CAP RBD manifest."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import logging
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_cap_paths import CHECKPOINT, PROCESSED_DIR, SLEEPFM_REPO
from sleepfm_cap_manifest_dataset import CapManifestEpochs


LOG = logging.getLogger(__name__)


def _resolve_encoder_class(qualified_name: str):
    search_roots = _sleepfm_search_roots()
    _ensure_sleepfm_namespace(search_roots)

    if qualified_name.lower() == "auto":
        class_obj, attempted = _auto_select_encoder_class(search_roots)
        if class_obj is not None:
            LOG.info(
                "Auto-detected encoder class %s from SleepFM repository",
                f"{class_obj.__module__}.{class_obj.__name__}",
            )
            return class_obj
        attempted_str = ", ".join(str(p) for p in attempted) if attempted else "<none>"
        raise ModuleNotFoundError(
            (
                "Failed to automatically locate a BAS encoder within the SleepFM repository. "
                "Specify --encoder-class explicitly. "
                f"Attempted files: {attempted_str}"
            )
        )

    module_path, class_name = qualified_name.rsplit(".", 1)
    module, attempted = _import_module_with_repo_fallback(module_path, search_roots)
    if module is not None:
        try:
            return getattr(module, class_name)
        except AttributeError:
            LOG.debug("Module %s imported but missing %s; falling back to search", module_path, class_name)

    class_obj, attempted_class_files = _load_class_from_repo(class_name, search_roots)
    if class_obj is not None:
        return class_obj

    attempted_all = attempted + attempted_class_files
    attempted_str = ", ".join(str(p) for p in attempted_all) if attempted_all else "<none>"
    raise ModuleNotFoundError(
        (
            f"Could not import '{module_path}'. "
            "Ensure the SleepFM repository is accessible at "
            f"{SLEEPFM_REPO} or install the package into the environment. "
            f"Attempted files: {attempted_str}"
        )
    )


def _sleepfm_search_roots() -> List[Path]:
    search_roots: List[Path] = []
    for candidate in (SLEEPFM_REPO, SLEEPFM_REPO / "src"):
        if candidate.exists():
            search_roots.append(candidate)
        else:
            LOG.debug("SleepFM path %s does not exist; skipping", candidate)
    sleepfm_pkg = SLEEPFM_REPO / "sleepfm"
    if sleepfm_pkg.exists():
        search_roots.append(sleepfm_pkg)
    return search_roots


def _ensure_sleepfm_namespace(search_roots: Sequence[Path]) -> None:
    """Ensure ``sleepfm`` behaves like a namespace package even without ``__init__``."""

    for root in search_roots:
        candidate = Path(root)
        if candidate.name == "sleepfm":
            pkg_root = candidate
            break
    else:
        pkg_root = SLEEPFM_REPO / "sleepfm"

    if not pkg_root.exists():
        return

    package_name = "sleepfm"
    if package_name not in sys.modules:
        module = types.ModuleType(package_name)
        module.__file__ = str(pkg_root / "__init__.py")
        module.__path__ = [str(pkg_root)]  # type: ignore[attr-defined]
        sys.modules[package_name] = module

    subpackages = {
        "sleepfm.models": pkg_root / "models",
        "sleepfm.encoders": pkg_root / "encoders",
    }
    for name, path in subpackages.items():
        if not path.exists():
            continue
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__file__ = str(path / "__init__.py")
            module.__path__ = [str(path)]  # type: ignore[attr-defined]
            module.__package__ = name.rsplit(".", 1)[0]
            sys.modules[name] = module


def _import_module_with_repo_fallback(
    module_path: str, search_roots: List[Path]
) -> Tuple[Optional[object], List[Path]]:
    """Import ``module_path`` with additional search paths inside ``SLEEPFM_REPO``."""

    try:
        return importlib.import_module(module_path), []
    except ModuleNotFoundError:
        pass

    injected = []
    for root in search_roots:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.append(root_str)
            injected.append(root_str)
    if injected:
        LOG.debug("Added SleepFM repo paths to sys.path: %s", injected)

    try:
        return importlib.import_module(module_path), []
    except ModuleNotFoundError:
        pass

    module, attempted = _load_module_from_repo(module_path, search_roots)
    return module, attempted


def _load_module_from_repo(module_path: str, search_roots: List[Path]) -> Tuple[Optional[object], List[Path]]:
    """Attempt to load ``module_path`` from explicit files inside ``search_roots``."""

    attempted: List[Path] = []
    rel_parts = module_path.split(".")
    target_stem = rel_parts[-1]

    for root in search_roots:
        candidate = root.joinpath(*rel_parts)
        direct_files = [candidate.with_suffix(".py"), candidate / "__init__.py"]
        for file_path in direct_files:
            attempted.append(file_path)
            module = _load_module_from_file(module_path, file_path)
            if module is not None:
                return module, attempted

    for root in search_roots:
        pattern = f"{target_stem}.py"
        for match in root.rglob(pattern):
            attempted.append(match)
            module_name = ".".join(match.relative_to(root).with_suffix("").parts)
            module = _load_module_from_file(module_name, match)
            if module is not None:
                if module_name != module_path:
                    sys.modules[module_path] = module
                return module, attempted

    return None, attempted


def _load_class_from_repo(class_name: str, search_roots: List[Path]) -> Tuple[Optional[type], List[Path]]:
    """Search for ``class_name`` inside ``search_roots`` and return the class if found."""

    attempted: List[Path] = []
    pattern = f"class {class_name}"

    for root in search_roots:
        for file_path in root.rglob("*.py"):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern not in text:
                continue
            attempted.append(file_path)
            module_name = ".".join(file_path.relative_to(root).with_suffix("").parts)
            module = _load_module_from_file(module_name, file_path)
            if module is None:
                continue
            attr = getattr(module, class_name, None)
            if attr is not None:
                return attr, attempted

    return None, attempted


def _auto_select_encoder_class(search_roots: List[Path]) -> Tuple[Optional[type], List[Path]]:
    """Attempt to automatically identify a BAS encoder class from SleepFM sources."""

    attempted: List[Path] = []
    preferred_names = [
        "BasEncoder",
        "BASencoder",
        "BASNet",
        "BasModel",
        "SleepFMBasEncoder",
        "SleepFMEncoder",
        "EfficientNet1D",
        "EfficientNetEncoder",
    ]

    for name in preferred_names:
        class_obj, tried = _load_class_from_repo(name, search_roots)
        attempted.extend(tried)
        if class_obj is not None:
            return class_obj, attempted

    try:
        from torch.nn import Module as TorchModule
    except ImportError:
        TorchModule = None  # type: ignore[assignment]

    if TorchModule is None:
        return None, attempted

    seen_files: Set[Path] = set()
    for root in search_roots:
        for file_path in root.rglob("*.py"):
            if file_path in seen_files:
                continue
            seen_files.add(file_path)
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "class" not in text or "Encoder" not in text:
                continue
            for match in re.finditer(r"class\s+(\w+)\s*\(([^)]*)\):", text):
                class_name = match.group(1)
                lower = class_name.lower()
                if "encoder" not in lower:
                    continue
                if not any(token in lower for token in ("bas", "sleepfm", "efficientnet")):
                    continue
                attempted.append(file_path)
                module_name = ".".join(file_path.relative_to(root).with_suffix("").parts)
                module = _load_module_from_file(module_name, file_path)
                if module is None:
                    continue
                attr = getattr(module, class_name, None)
                if attr is None or not inspect.isclass(attr):
                    continue
                try:
                    if issubclass(attr, TorchModule):
                        return attr, attempted
                except TypeError:
                    continue

    return None, attempted


def _load_module_from_file(module_name: str, file_path: Path) -> Optional[object]:
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except ModuleNotFoundError as exc:
        LOG.debug("Failed to execute module %s from %s: %s", module_name, file_path, exc)
        return None
    return module


def _collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tensors = torch.stack([item["bas"] for item in batch])
    return {
        "bas": tensors,
        "stage_code": [item["stage_code"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "epoch_idx": [item["epoch_idx"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def _load_manifest_subset(
    manifest_path: Path,
    output_csv: Path,
    embeddings_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Set[str]]:
    """Return (manifest, existing_embeddings, pending_manifest, completed_subjects)."""

    manifest = pd.read_csv(manifest_path)
    existing = (
        pd.read_csv(output_csv)
        if output_csv.exists()
        else pd.DataFrame(columns=["subject_id", "epoch_idx", "stage_code", "emb_path"])
    )

    completed_subjects: Set[str] = set()
    for subject_dir in embeddings_dir.glob("*/"):
        marker = subject_dir / ".done"
        subject_id = subject_dir.name
        if marker.exists():
            completed_subjects.add(subject_id)

    if not existing.empty:
        existing_pairs = existing[["subject_id", "epoch_idx"]].drop_duplicates()
        merged = manifest.merge(existing_pairs.assign(_present=True), how="left", on=["subject_id", "epoch_idx"])
        pending = merged[merged["_present"].isna()].drop(columns=["_present"])
    else:
        pending = manifest.copy()

    if completed_subjects:
        pending = pending[~pending["subject_id"].isin(completed_subjects)]

    return manifest, existing, pending.reset_index(drop=True), completed_subjects


def generate_embeddings(
    manifest_path: Path,
    output_csv: Path,
    encoder_class_path: str,
    checkpoint_path: Path,
    checkpoint_key: Optional[str],
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> None:
    for candidate in (SLEEPFM_REPO, SLEEPFM_REPO / "src"):
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)

    encoder_class = _resolve_encoder_class(encoder_class_path)
    LOG.info(
        "Initialising encoder class %s.%s",
        encoder_class.__module__,
        encoder_class.__name__,
    )
    try:
        model = encoder_class()
    except TypeError as exc:
        raise TypeError(
            "Failed to instantiate the SleepFM encoder. "
            "If the class requires constructor arguments, provide a wrapper "
            "or adjust --encoder-class to reference a zero-argument factory."
        ) from exc
    state = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key:
        state = state[checkpoint_key]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    embeddings_dir = output_csv.parent / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    manifest, existing, pending, completed_subjects = _load_manifest_subset(
        manifest_path,
        output_csv,
        embeddings_dir,
    )

    if pending.empty:
        LOG.info(
            "No pending epochs detected. %d subjects already completed (markers: %s)",
            len(completed_subjects),
            ", ".join(sorted(completed_subjects)) if completed_subjects else "none",
        )
        return

    dataset = CapManifestEpochs(manifest_path, frame=pending)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_batch,
    )

    records: List[Dict[str, object]] = []
    processed_subjects: Set[str] = set()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding BAS epochs"):
            bas = batch["bas"].to(device=device, non_blocking=True)
            outputs = model(bas)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            embeddings = outputs.detach().cpu().numpy()
            for idx, vector in enumerate(embeddings):
                subject_id = batch["subject_id"][idx]
                epoch_idx = batch["epoch_idx"][idx]
                processed_subjects.add(subject_id)
                emb_dir = embeddings_dir / subject_id
                emb_dir.mkdir(parents=True, exist_ok=True)
                emb_path = emb_dir / f"epoch_{epoch_idx:06d}.npy"
                np.save(emb_path, vector.astype(np.float32), allow_pickle=False)
                records.append(
                    {
                        "subject_id": subject_id,
                        "epoch_idx": epoch_idx,
                        "stage_code": batch["stage_code"][idx],
                        "emb_path": emb_path.as_posix(),
                    }
                )

    if not records:
        LOG.info("No new embeddings were generated.")
        return

    new_frame = pd.DataFrame.from_records(records)
    combined = pd.concat([existing, new_frame], ignore_index=True)
    combined.sort_values(["subject_id", "epoch_idx"], inplace=True)
    combined.to_csv(output_csv, index=False)

    for subject_id in processed_subjects:
        subject_manifest = manifest[manifest["subject_id"] == subject_id]
        subject_embeddings = combined[combined["subject_id"] == subject_id]
        marker_path = embeddings_dir / subject_id / ".done"
        if len(subject_embeddings) >= len(subject_manifest):
            marker_path.touch()
            LOG.info("Marked subject %s as complete (embeddings: %d)", subject_id, len(subject_embeddings))
        else:
            if marker_path.exists():
                marker_path.unlink()
            LOG.info(
                "Subject %s has %d/%d embeddings; leaving marker absent",
                subject_id,
                len(subject_embeddings),
                len(subject_manifest),
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "manifest_cap_rbd_bas_only.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "embeddings_bas.csv")
    parser.add_argument(
        "--encoder-class",
        type=str,
        default="auto",
        help="Fully-qualified encoder class name or 'auto' to detect the BAS encoder",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default="bas_encoder",
        help="State-dict key containing the encoder weights (use empty string to skip)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoint_key = args.checkpoint_key or None
    device = torch.device(args.device)
    generate_embeddings(
        manifest_path=args.manifest,
        output_csv=args.output,
        encoder_class_path=args.encoder_class,
        checkpoint_path=args.checkpoint,
        checkpoint_key=checkpoint_key,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

