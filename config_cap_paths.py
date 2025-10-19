"""Configuration for CAP RBD to SleepFM pipeline paths.

This module centralises the filesystem locations required by the
preprocessing, embedding generation, and probe training scripts. Update the
paths to match the local environment before running the pipeline.
"""

from pathlib import Path

RAW_DIR = Path(r"C:\\Users\\JD\\sleepfm_cap_rbd\\data\\raw")
PROCESSED_DIR = Path(r"C:\\Users\\JD\\sleepfm_cap_rbd\\sleepfm_processed")
SLEEPFM_REPO = Path(r"C:\\Users\\JD\\sleepfm_cap_rbd\\sleepfm-codebase")
CHECKPOINT = Path(r"C:\\Users\\JD\\Downloads\\best.pt")

FS_TARGET = 256
EPOCH_LEN_S = 30

