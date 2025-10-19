# CAP-RBD SleepFM Integration Toolkit

This repository contains utility scripts for converting the CAP REM Sleep Behavior Disorder (RBD) subset of the CAP Sleep Database into the format expected by the SleepFM foundation model, generating embeddings, and training lightweight probes for downstream analysis.

## Repository Contents

| Script | Summary |
| --- | --- |
| `config_cap_paths.py` | Central location for file-system paths and global preprocessing constants (sampling rate and epoch duration). Import this module from other scripts to ensure all stages reference the same directories. |
| `bridge_cap_to_sleepfm.py` | Parses CAP RBD EDF recordings and their TXT annotations, standardises the selected BAS channels (EEG/EOG/EMG), segments them into 30-second epochs, saves each epoch as a `.npy` shard in SleepFM's directory layout, and emits a manifest CSV describing every epoch. |
| `sleepfm_cap_manifest_dataset.py` | Provides a PyTorch `Dataset` (`CapManifestEpochs`) that consumes the manifest CSV produced by the bridge script and lazily loads BAS epoch tensors alongside metadata (stage codes, subject IDs, epoch indices). |
| `generate_embeddings_bas.py` | Loads the pretrained SleepFM BAS encoder weights, iterates over epochs via `CapManifestEpochs`, and stores embedding vectors and an accompanying index CSV for later tasks. The script now auto-detects a suitable BAS encoder class from the SleepFM source tree when `--encoder-class` is left at its default (`auto`). |
| `train_rbd_probe.py` | Trains logistic-regression probes on the saved embeddings to evaluate sleep-stage classification and REM-with-abnormal-muscle-activity detection, reporting cross-validated AUROC/AUPRC scores. |

## How the Pieces Fit Together

The end-to-end pipeline is executed sequentially:

1. **Preprocess raw CAP data** with `bridge_cap_to_sleepfm.py`. This script depends on `config_cap_paths.py` for locating the raw EDF/TXT files (`RAW_DIR`) and the SleepFM-style processed directory (`PROCESSED_DIR`). It outputs per-subject epoch shards under `sleepfm_processed/shards/<subject>/BAS/` and a manifest called `manifest_cap_rbd_bas_only.csv` summarising all epochs, including stage codes and file paths.
2. **Load epochs for inference** using `sleepfm_cap_manifest_dataset.py`. This dataset implementation is reused by `generate_embeddings_bas.py` to feed batches into the encoder.
3. **Generate embeddings** with `generate_embeddings_bas.py`. The script loads the pretrained SleepFM BAS encoder (`CHECKPOINT`) and writes 30-second epoch embeddings to `sleepfm_processed/embeddings/`, while producing an index file `embeddings_bas.csv`. The manifest and embedding index share subject IDs and epoch indices, enabling downstream merges.
4. **Train evaluation probes** via `train_rbd_probe.py`. It merges the manifest and embedding indices, constructs sleep-stage and proxy RBD labels, performs subject-level cross-validation, and prints metrics to stdout.

Each stage shares configuration through `config_cap_paths.py`, ensuring consistent target sampling frequency (`FS_TARGET = 256 Hz`) and epoch length (`EPOCH_LEN_S = 30 s`) across the pipeline.

## Assumptions and Requirements

* **Channel availability**: The bridge script assumes the BAS channel bundle (EEG, EOG, EMG) can be extracted from each EDF. Missing channels will trigger warnings or skips depending on severity.
* **Annotation format**: CAP RBD TXT annotations must align with the helper parsing logic in `bridge_cap_to_sleepfm.py`. They should provide sleep stages and any REM/EMG events needed for RBD proxy labelling.
* **Checkpoint format**: `generate_embeddings_bas.py` expects a pretrained SleepFM checkpoint that contains BAS encoder weights accessible via the key used in the script (e.g., `"bas_encoder"`). Adjust loading code if the checkpoint structure differs.
* **Dependencies**: All scripts require the SleepFM repository dependencies plus additional libraries (`mne`, `numpy`, `torch`, `pandas`, `scikit-learn`). Install them in the same environment as outlined in the original project instructions.

## Usage Guide

1. **Configure paths**: Edit `config_cap_paths.py` so that `RAW_DIR`, `PROCESSED_DIR`, `SLEEPFM_REPO`, and `CHECKPOINT` match your local machine.
2. **Activate environment**: Use the Python environment that contains SleepFM and the required dependencies.
3. **Run preprocessing**:
   ```bash
   python bridge_cap_to_sleepfm.py
   ```
4. **Generate embeddings** (after preprocessing completes):
   ```bash
   python generate_embeddings_bas.py
   ```
   The command attempts to auto-detect the BAS encoder definition from the SleepFM repository referenced in `config_cap_paths.py`. If you maintain a custom checkout or renamed modules, override the discovery step with `--encoder-class <module.ClassName>`.
5. **Train probes**:
   ```bash
   python train_rbd_probe.py
   ```

Intermediate outputs (manifests, shard directories, embeddings) are stored under `PROCESSED_DIR`, making the pipeline restartable from any stage by reusing the generated artifacts.

### Incremental and Notebook-Friendly Execution

* **Automatic resume points**: Each time `bridge_cap_to_sleepfm.py` finishes a subject successfully it creates a `.done` marker inside `sleepfm_processed/shards/<subject_id>/`. When you rerun the script it will skip subjects that already have both the marker and their BAS shard files, so you can safely resume after an interruption without reprocessing earlier nights. If shards are present but the marker is missing (e.g., from an earlier run of the script), the bridge now recognises the completed subject, creates the marker, and skips ahead automatically. To force a redo for a particular subject, delete its `.done` file (and optionally the shard directory) before re-running the script.
* **Manifest de-duplication**: Reprocessing a subject replaces its rows inside `manifest_cap_rbd_bas_only.csv`, keeping the manifest consistent even after partial reruns.
* **Embedding resume support**: `generate_embeddings_bas.py` inspects the existing `embeddings_bas.csv` file and the `.done` markers under `sleepfm_processed/embeddings/<subject_id>/`. Any epoch that already has an embedding on disk is skipped, so you can restart the embedding generation stage and it will pick up exactly where it left off. The script only recomputes embeddings for missing `(subject_id, epoch_idx)` pairs and appends them to the CSV.
* **Running from Jupyter**: When your notebook lives beside these scripts, you can execute the full pipeline from a single cell without changing directories. A minimal cell looks like this:

  ```python
  import subprocess

  commands = [
      ["python", "bridge_cap_to_sleepfm.py"],
      ["python", "generate_embeddings_bas.py"],
      ["python", "train_rbd_probe.py"],
  ]

  for cmd in commands:
      print(f"Running: {' '.join(cmd)}")
      completed = subprocess.run(cmd, check=True)
  ```

  Because the scripts create `.done` markers and skip completed work, re-running the cell resumes from the first unfinished subject or embedding batch instead of recomputing everything.

## Extending the Toolkit

* Add multi-modality support by modifying `bridge_cap_to_sleepfm.py` to emit ECG or respiratory modalities and adjusting SleepFM configuration accordingly.
* Swap the probe model in `train_rbd_probe.py` for alternative classifiers (e.g., linear SVM, gradient boosting) to compare performance.
* Use the embeddings and manifests as inputs to downstream analytics or visualisation tools, such as event-level attribution studies or SleepFM fine-tuning scripts.

For additional project context—including environment setup, pipeline execution order, and quality checks—refer back to the original instruction document supplied with the dataset.
