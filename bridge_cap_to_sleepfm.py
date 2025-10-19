"""Bridge CAP RBD recordings into SleepFM-compatible epoch shards.

The script ingests raw EDF polysomnography files alongside the CAP text
annotations, converts them into 30-second BAS epochs resampled at 256 Hz,
and generates a manifest CSV linking each epoch to its metadata. The output is
organised in the layout expected by SleepFM:

```
<processed_root>/
  ├── shards/<subject_id>/BAS/epoch_XXXXXX.npy
  └── manifest_cap_rbd_bas_only.csv
```

Example usage
-------------
    python bridge_cap_to_sleepfm.py \
        --raw-dir /path/to/raw \
        --processed-dir /path/to/sleepfm_processed

Before running the script, edit :mod:`config_cap_paths` so that it reflects the
local environment, or pass the desired paths as CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import mne
import numpy as np
import pandas as pd

from config_cap_paths import EPOCH_LEN_S, FS_TARGET, PROCESSED_DIR, RAW_DIR

LOGGER = logging.getLogger(__name__)

DONE_MARKER_NAME = ".done"


@dataclass
class StageInterval:
    """Stores a single sleep-stage interval."""

    start_s: float
    end_s: float
    stage_code: str


@dataclass
class EmgEvent:
    """Stores an abnormal EMG activity interval."""

    start_s: float
    end_s: float


STAGE_ALIASES: Dict[str, str] = {
    "w": "W",
    "wake": "W",
    "n1": "N1",
    "s1": "N1",
    "n2": "N2",
    "s2": "N2",
    "n3": "N3",
    "s3": "N3",
    "s4": "N3",
    "rem": "REM",
    "r": "REM",
    "?": "UNKNOWN",
    "movement": "UNKNOWN",
}

EVENT_KEYWORDS = ("rbd", "emg", "muscle", "phasic", "tonic")


def _normalise_stage_label(label: str) -> Optional[str]:
    label_norm = label.strip().lower()
    if label_norm in STAGE_ALIASES:
        return STAGE_ALIASES[label_norm]
    label_upper = label.strip().upper()
    if label_upper in {"W", "N1", "N2", "N3", "REM", "UNKNOWN"}:
        return label_upper
    if label_upper in {"N4", "S4"}:
        return "N3"
    return None


def parse_txt_annotations(annotation_path: Path) -> Tuple[List[StageInterval], List[EmgEvent]]:
    """Parse a CAP text annotation file into stage intervals and EMG events.

    The CAP RBD text files are not fully standardised, therefore the parser is
    intentionally tolerant: it attempts to extract timestamps in ``HH:MM:SS``
    format (optionally with decimal seconds) and then interprets the remaining
    tokens as either stage labels or event descriptions. Any line that contains
    keywords associated with abnormal EMG activity is treated as an event.
    """

    time_pattern = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}(?:\.\d+)?)")
    stage_intervals: List[StageInterval] = []
    emg_events: List[EmgEvent] = []

    with annotation_path.open("r", encoding="utf-8", errors="ignore") as handle:
        previous_stage: Optional[StageInterval] = None
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = time_pattern.search(line)
            if not match:
                continue
            start_s = (
                int(match.group("h")) * 3600
                + int(match.group("m")) * 60
                + float(match.group("s"))
            )
            trailing = line[match.end() :].strip()
            tokens = re.split(r"[\s;:,]+", trailing)
            tokens = [tok for tok in tokens if tok]
            stage_label: Optional[str] = None
            if tokens:
                stage_label = _normalise_stage_label(tokens[0])
            if stage_label:
                if previous_stage is not None:
                    previous_stage.end_s = start_s
                    stage_intervals.append(previous_stage)
                previous_stage = StageInterval(start_s=start_s, end_s=start_s, stage_code=stage_label)
                continue
            lower_line = trailing.lower()
            if any(keyword in lower_line for keyword in EVENT_KEYWORDS):
                duration_match = re.search(r"duration[=:\s]+(\d+(?:\.\d+)?)", lower_line)
                duration_s = float(duration_match.group(1)) if duration_match else 0.0
                end_s = start_s + duration_s if duration_s > 0 else start_s + 30.0
                emg_events.append(EmgEvent(start_s=start_s, end_s=end_s))
        if previous_stage is not None:
            stage_intervals.append(previous_stage)

    stage_intervals.sort(key=lambda interval: interval.start_s)
    emg_events.sort(key=lambda event: event.start_s)
    return stage_intervals, emg_events


def build_epoch_hypnogram(
    stage_intervals: Sequence[StageInterval],
    n_epochs: int,
    epoch_len_s: float,
) -> List[str]:
    """Assign a stage code to each epoch."""

    hypnogram = ["UNKNOWN"] * n_epochs
    for interval in stage_intervals:
        start_epoch = max(int(interval.start_s // epoch_len_s), 0)
        end_epoch = min(int(math.ceil(interval.end_s / epoch_len_s)), n_epochs)
        for epoch in range(start_epoch, end_epoch):
            hypnogram[epoch] = interval.stage_code
    return hypnogram


def build_epoch_emg_mask(
    emg_events: Sequence[EmgEvent],
    n_epochs: int,
    epoch_len_s: float,
) -> List[bool]:
    """Mark epochs that overlap with an abnormal EMG event."""

    mask = [False] * n_epochs
    for event in emg_events:
        start_epoch = max(int(event.start_s // epoch_len_s), 0)
        end_epoch = min(int(math.ceil(event.end_s / epoch_len_s)), n_epochs)
        for epoch in range(start_epoch, end_epoch):
            mask[epoch] = True
    return mask


def _pick_bas_channels(raw: mne.io.BaseRaw) -> Tuple[List[int], List[str]]:
    """Select EEG, EOG, and EMG channels."""

    picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True, ecg=False)
    if picks.size == 0:
        raise RuntimeError("No EEG/EOG/EMG channels found in the recording")
    channel_names = [raw.ch_names[idx] for idx in picks]
    return picks, channel_names


def _zscore(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0.0] = 1.0
    return (data - mean) / std


def _subject_paths(processed_root: Path, subject_id: str) -> Tuple[Path, Path]:
    """Return (BAS shard directory, done marker path) for the given subject."""

    subject_root = processed_root / "shards" / subject_id
    bas_dir = subject_root / "BAS"
    done_marker = subject_root / DONE_MARKER_NAME
    return bas_dir, done_marker


def process_recording(
    edf_path: Path,
    annotation_path: Path,
    processed_root: Path,
    epoch_len_s: float = EPOCH_LEN_S,
    target_fs: int = FS_TARGET,
) -> Tuple[int, int]:
    """Process a single recording into epoch shards and return summary stats."""

    subject_id = edf_path.stem
    LOGGER.info("Processing %s", subject_id)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    picks, channel_names = _pick_bas_channels(raw)
    raw.pick(picks)
    raw.resample(target_fs)

    data = raw.get_data().astype(np.float32)
    data = _zscore(data)
    samples_per_epoch = int(target_fs * epoch_len_s)
    n_epochs = data.shape[1] // samples_per_epoch
    if n_epochs == 0:
        raise RuntimeError(f"Recording {edf_path} is too short for even a single epoch")
    total_samples = n_epochs * samples_per_epoch
    data = data[:, :total_samples]

    stage_intervals, emg_events = parse_txt_annotations(annotation_path)
    hypnogram = build_epoch_hypnogram(stage_intervals, n_epochs, epoch_len_s)
    emg_mask = build_epoch_emg_mask(emg_events, n_epochs, epoch_len_s)

    subject_dir, done_marker = _subject_paths(processed_root, subject_id)
    subject_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for epoch_idx in range(n_epochs):
        start = epoch_idx * samples_per_epoch
        end = start + samples_per_epoch
        epoch_array = data[:, start:end]
        epoch_path = subject_dir / f"epoch_{epoch_idx:06d}.npy"
        np.save(epoch_path, epoch_array, allow_pickle=False)

        t0 = epoch_idx * epoch_len_s
        t1 = t0 + epoch_len_s
        manifest_rows.append(
            {
                "subject_id": subject_id,
                "epoch_idx": epoch_idx,
                "t0_s": t0,
                "t1_s": t1,
                "stage_code": hypnogram[epoch_idx],
                "bas_path": epoch_path.as_posix(),
                "ecg_path": "",
                "resp_path": "",
                "bas_channels_json": json.dumps(channel_names),
                "rbd_event": bool(emg_mask[epoch_idx]),
            }
        )

    manifest_path = processed_root / "manifest_cap_rbd_bas_only.csv"
    new_rows = pd.DataFrame(manifest_rows)
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        existing = existing[existing["subject_id"] != subject_id]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.sort_values(["subject_id", "epoch_idx"], inplace=True)
    combined.to_csv(manifest_path, index=False)

    done_marker.touch()
    LOGGER.info("%s → %d epochs", subject_id, n_epochs)
    return n_epochs, data.shape[0]


def discover_recordings(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """Return a list of (edf, annotation) path pairs."""

    pairs: List[Tuple[Path, Path]] = []
    for edf_path in sorted(raw_dir.glob("*.edf")):
        annotation_path = edf_path.with_suffix(".txt")
        if not annotation_path.exists():
            LOGGER.warning("Missing annotation file for %s", edf_path.name)
            continue
        pairs.append((edf_path, annotation_path))
    return pairs


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory containing EDF/TXT pairs")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory where SleepFM-formatted data will be written",
    )
    parser.add_argument("--epoch-length", type=float, default=EPOCH_LEN_S, help="Epoch length in seconds")
    parser.add_argument("--target-fs", type=int, default=FS_TARGET, help="Target sampling rate (Hz)")
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Overwrite any existing manifest instead of appending",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    processed_root = args.processed_dir
    processed_root.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_root / "manifest_cap_rbd_bas_only.csv"
    if args.overwrite_manifest and manifest_path.exists():
        manifest_path.unlink()

    manifest_frame: Optional[pd.DataFrame]
    if manifest_path.exists():
        manifest_frame = pd.read_csv(manifest_path)
    else:
        manifest_frame = None

    recording_pairs = discover_recordings(args.raw_dir)
    if not recording_pairs:
        LOGGER.warning("No EDF/TXT pairs found in %s", args.raw_dir)
        return

    total_epochs = 0
    channel_counts: List[int] = []
    processed_this_run: Set[str] = set()
    for edf_path, annotation_path in recording_pairs:
        subject_id = edf_path.stem
        subject_dir, done_marker = _subject_paths(processed_root, subject_id)
        shard_exists = subject_dir.exists() and any(subject_dir.glob("*.npy"))
        subject_manifest = None
        if manifest_frame is not None:
            subject_manifest = manifest_frame[manifest_frame["subject_id"] == subject_id]
        if done_marker.exists():
            if shard_exists:
                if subject_manifest is not None and not subject_manifest.empty:
                    LOGGER.info(
                        "Skipping %s because %s is present and shards already exist",
                        subject_id,
                        done_marker,
                    )
                    total_epochs += len(subject_manifest)
                    try:
                        first_channels = json.loads(subject_manifest.iloc[0]["bas_channels_json"])
                        channel_counts.append(len(first_channels))
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                LOGGER.warning(
                    "Detected %s for %s but manifest rows are missing; regenerating subject",
                    done_marker,
                    subject_id,
                )
            else:
                LOGGER.warning(
                    "Found %s for %s but no shards detected; removing marker and reprocessing",
                    done_marker,
                    subject_id,
                )
            done_marker.unlink(missing_ok=True)
        elif shard_exists:
            if subject_manifest is None or subject_manifest.empty:
                LOGGER.info(
                    "Detected existing shards for %s but no manifest rows found; reprocessing",
                    subject_id,
                )
            else:
                LOGGER.info(
                    "Detected existing shards for %s without a %s marker; assuming prior success",
                    subject_id,
                    DONE_MARKER_NAME,
                )
                done_marker.touch()
                total_epochs += len(subject_manifest)
                try:
                    first_channels = json.loads(subject_manifest.iloc[0]["bas_channels_json"])
                    channel_counts.append(len(first_channels))
                except Exception:  # noqa: BLE001
                    pass
                continue
        try:
            epochs, n_channels = process_recording(
                edf_path,
                annotation_path,
                processed_root,
                epoch_len_s=args.epoch_length,
                target_fs=args.target_fs,
            )
        except Exception as error:  # noqa: BLE001 - provide context when a file fails
            LOGGER.error("Failed to process %s: %s", edf_path.name, error)
            continue
        total_epochs += epochs
        channel_counts.append(n_channels)
        processed_this_run.add(subject_id)
        # Reload manifest so subsequent iterations have access to the fresh rows
        manifest_frame = pd.read_csv(manifest_path)

    if not channel_counts and manifest_path.exists():
        manifest_frame = pd.read_csv(manifest_path)
        if not manifest_frame.empty:
            channel_counts = [
                len(json.loads(row))
                for row in manifest_frame.drop_duplicates("subject_id")["bas_channels_json"]
            ]
            total_epochs = len(manifest_frame)

    if processed_this_run:
        LOGGER.info("Processed %d subjects in this run", len(processed_this_run))
    elif channel_counts:
        LOGGER.info("No subjects required processing in this run; using existing manifest summary")

    LOGGER.info("Subjects with available BAS shards: %d", len(channel_counts))
    LOGGER.info("Total epochs: %d", total_epochs)
    if channel_counts:
        LOGGER.info(
            "Channel count (min/median/max): %d / %d / %d",
            min(channel_counts),
            int(np.median(channel_counts)),
            max(channel_counts),
        )


if __name__ == "__main__":
    main()

