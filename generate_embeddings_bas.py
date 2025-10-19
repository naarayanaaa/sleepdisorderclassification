"""Generate SleepFM BAS embeddings for the CAP RBD manifest."""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_cap_paths import CHECKPOINT, PROCESSED_DIR, SLEEPFM_REPO
from sleepfm_cap_manifest_dataset import CapManifestEpochs


LOG = logging.getLogger(__name__)


def _resolve_encoder_class(qualified_name: str):
    module_path, class_name = qualified_name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        candidate_paths = [SLEEPFM_REPO, SLEEPFM_REPO / "src"]
        injected = []
        for candidate in candidate_paths:
            if candidate.exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.append(candidate_str)
                    injected.append(candidate_str)
        if injected:
            LOG.debug("Added SleepFM repo paths to sys.path: %s", injected)
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as retry_error:
            raise ModuleNotFoundError(
                (
                    f"Could not import '{module_path}'. "
                    "Verify that the SleepFM repository is available at "
                    f"{SLEEPFM_REPO} or install the package into the environment."
                )
            ) from retry_error
    return getattr(module, class_name)


def _collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tensors = torch.stack([item["bas"] for item in batch])
    return {
        "bas": tensors,
        "stage_code": [item["stage_code"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "epoch_idx": [item["epoch_idx"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


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
    model = encoder_class()
    state = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key:
        state = state[checkpoint_key]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = CapManifestEpochs(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_batch,
    )

    embeddings_dir = output_csv.parent / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
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

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "epoch_idx", "stage_code", "emb_path"])
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROCESSED_DIR / "manifest_cap_rbd_bas_only.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "embeddings_bas.csv")
    parser.add_argument(
        "--encoder-class",
        type=str,
        default="sleepfm.models.bas_encoder.BasEncoder",
        help="Fully-qualified encoder class name",
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

