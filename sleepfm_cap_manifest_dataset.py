"""Dataset utilities for CAP RBD manifest-driven BAS epochs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class EpochRecord:
    """Represents a single BAS epoch entry loaded from the manifest."""

    bas_path: Path
    stage_code: str
    subject_id: str
    epoch_idx: int
    metadata: Dict[str, Any]


class CapManifestEpochs(Dataset):
    """Loads BAS epochs listed in the CAP manifest CSV.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest_cap_rbd_bas_only.csv`` (or an equivalent file).
    columns:
        Additional manifest columns to keep as metadata. If ``None`` all columns
        beyond the required minimum will be preserved.
    dtype:
        ``torch.dtype`` to cast the loaded arrays to (default: ``torch.float32``).
    device:
        Optional device where the tensors should be placed.
    """

    required_columns = {"bas_path", "stage_code", "subject_id", "epoch_idx"}

    def __init__(
        self,
        manifest_path: Path,
        columns: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        frame: Optional[pd.DataFrame] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if frame is None:
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            source_frame = pd.read_csv(self.manifest_path)
        else:
            source_frame = frame.copy()
        self._frame = source_frame.reset_index(drop=True)
        missing = self.required_columns.difference(self._frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        self._dtype = dtype
        self._device = device

        if columns is None:
            metadata_columns = [
                col for col in self._frame.columns if col not in self.required_columns
            ]
        else:
            metadata_columns = list(columns)
        self._metadata_columns = metadata_columns

    def __len__(self) -> int:  # noqa: D401 - standard Dataset interface
        return len(self._frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self._frame.iloc[index]
        bas_path = Path(row["bas_path"])
        if not bas_path.exists():
            raise FileNotFoundError(f"Epoch file not found: {bas_path}")
        array = np.load(bas_path)
        tensor = torch.as_tensor(array, dtype=self._dtype)
        if self._device is not None:
            tensor = tensor.to(self._device)

        metadata = {col: row[col] for col in self._metadata_columns}
        return {
            "bas": tensor,
            "stage_code": row["stage_code"],
            "subject_id": row["subject_id"],
            "epoch_idx": int(row["epoch_idx"]),
            "metadata": metadata,
        }

