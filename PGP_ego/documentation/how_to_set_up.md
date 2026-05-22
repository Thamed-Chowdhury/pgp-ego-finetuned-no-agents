# PGP Ego — Setup and Usage Guide

This guide covers everything needed to reproduce the ego vehicle trajectory prediction pipeline from scratch: environment setup, data download, preprocessing, training, fine-tuning, evaluation, and visualization.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Environment Setup](#3-environment-setup)
4. [nuScenes Data Download](#4-nuscenes-data-download)
5. [Directory Structure](#5-directory-structure)
6. [Preprocessing](#6-preprocessing)
7. [Training from Scratch](#7-training-from-scratch)
8. [Fine-tuning from a Checkpoint](#8-fine-tuning-from-a-checkpoint)
9. [Evaluation](#9-evaluation)
10. [Visualization](#10-visualization)
11. [Config File Reference](#11-config-file-reference)
12. [Model Architecture](#12-model-architecture)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Overview

`PGP_ego` extends the [PGP (Prediction via Graph-based Policy)](https://github.com/nachiket92/PGP) trajectory prediction model to predict the ego vehicle's future trajectory. The original PGP predicts other annotated agents (cars, pedestrians). This extension treats the ego vehicle as an additional prediction target by building a custom dataset class (`NuScenesEgoGraphs`) that reads ego poses from nuScenes' `ego_pose` table and constructs the same graph-format inputs that PGP expects.

**What stays the same:** Model architecture (PGP Encoder + PGP Aggregator + LVM Decoder), training loop, preprocessing pipeline, all config formats.

**What is new:**
- `datasets/nuScenes/nuScenes_ego_graphs.py` — ego dataset class
- `configs/pgp_ego_gatx2_lvm_traversal.yml` — ego training config
- `configs/preprocess_nuscenes_ego.yml` — ego preprocessing config
- `visualize_ego_scenes.py` — 3-panel scene visualization
- `eval_top_confidence.py` — top-confidence ADE/FDE evaluation

---

## 2. Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Preprocessing | CPU-only, 8 GB RAM | 16 GB RAM (nuScenes metadata = ~13 GB when fully loaded) |
| Training | NVIDIA GPU, 8 GB VRAM | NVIDIA T4 (16 GB VRAM) or better |
| Inference / Visualization | GPU optional (CPU works) | GPU for speed |
| Evaluation (batch) | GPU, 6 GB VRAM | T4 |

> **RAM note**: NuScenes `v1.0-trainval` loads ~13.4 GB of metadata into RAM at startup. You need at least 16 GB total RAM to run any script that initialises `NuScenes(version='v1.0-trainval', ...)`.

---

## 3. Environment Setup

### 3.1 Clone the repository

```bash
git clone https://github.com/nachiket92/PGP.git PGP_ego
cd PGP_ego
```

Or if you already have the directory:

```bash
cd /path/to/PGP_ego
```

### 3.2 Create a conda environment

```bash
conda create -n pgp_ego python=3.12 -y
conda activate pgp_ego
```

### 3.3 Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust for your CUDA version
pip install nuscenes-devkit==1.2.0
pip install ray==2.55.0
pip install positional-encodings==6.0.4
pip install pyquaternion matplotlib pyyaml tensorboard imageio
```

**Package versions used in this project:**

| Package | Version |
|---------|---------|
| torch | 2.8.0+cu128 |
| nuscenes-devkit | 1.2.0 |
| ray | 2.55.0 |
| positional-encodings | 6.0.4 |

### 3.4 Fix positional-encodings import (v6 API change)

If you see `ImportError: cannot import name 'PositionalEncoding1D' from 'positional_encodings'`, the import path changed in v6:

```python
# Old (v5 and earlier)
from positional_encodings import PositionalEncoding1D

# New (v6+)
from positional_encodings.torch_encodings import PositionalEncoding1D
```

This fix is already applied in `PGP_ego/models/encoders/pgp_encoder.py`.

---

## 4. nuScenes Data Download

PGP works on **metadata and maps only** — no sensor data (camera, LiDAR, RADAR) is needed.

### 4.1 Required packages (~290 MB total)

**Package 1: Metadata** (~130 MB compressed)

Download from the [nuScenes website](https://www.nuscenes.org/nuscenes#download) (requires free account):
- `v1.0-trainval_meta.tgz`

```bash
tar -xzf v1.0-trainval_meta.tgz -C /path/to/nuscenes_data/
```

**Package 2: Map expansion** (~381 MB)

```bash
wget https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip
unzip nuScenes-map-expansion-v1.3.zip -d /path/to/nuscenes_data/
```

> The map zip extracts without a `maps/` prefix. If the files land directly (e.g., `expansion/`, `basemap/`, `*.png`), move them into `maps/`:
> ```bash
> mkdir -p /path/to/nuscenes_data/maps
> mv expansion basemap prediction *.png /path/to/nuscenes_data/maps/
> ```

### 4.2 What is NOT needed

| Package | Size | Why skippable |
|---------|------|---------------|
| `v1.0-trainval01_blobs.tgz` … `10_blobs.tgz` | ~300 GB | Camera / LiDAR / RADAR — never read by PGP |
| `samples/`, `sweeps/` | ~300 GB | PGP only reads bounding-box annotations, never file paths |

### 4.3 Verify data layout

```
nuscenes_data/
├── v1.0-trainval/
│   ├── sample_annotation.json
│   ├── sample.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── scene.json
│   ├── log.json
│   └── ... (7 more JSON files)
└── maps/
    ├── singapore-onenorth.json       ← NOT here; these are in expansion/
    ├── expansion/
    │   ├── singapore-onenorth.json
    │   ├── singapore-hollandvillage.json
    │   ├── singapore-queenstown.json
    │   └── boston-seaport.json
    ├── prediction/
    │   └── prediction_scenes.json
    ├── basemap/
    └── *.png  (4 raster map images)
```

---

## 5. Directory Structure

```
PGP_ego/
├── configs/
│   ├── preprocess_nuscenes_ego.yml       ← ego preprocessing config
│   ├── pgp_ego_gatx2_lvm_traversal.yml  ← ego training config (full)
│   ├── pgp_ego_smoketest.yml             ← smoke test config (2 epochs)
│   ├── preprocess_nuscenes.yml           ← agent preprocessing config
│   └── pgp_gatx2_lvm_traversal.yml      ← agent training config
│
├── datasets/nuScenes/
│   ├── nuScenes_ego_graphs.py            ← NEW: ego dataset class
│   ├── nuScenes_graphs.py                ← parent class (graph representation)
│   ├── nuScenes_vector.py                ← grandparent class (vector features)
│   └── nuScenes.py                       ← base nuScenes dataset
│
├── models/
│   ├── encoders/pgp_encoder.py           ← GAT-based encoder
│   ├── aggregators/pgp.py                ← lane-graph traversal aggregator
│   └── decoders/lvm.py                   ← latent variable model decoder
│
├── train_eval/
│   ├── trainer.py                        ← training loop
│   ├── evaluator.py                      ← evaluation loop
│   ├── preprocessor.py                   ← preprocessing orchestrator
│   └── initialization.py                 ← factory functions
│
├── preprocess.py                         ← preprocessing entry point
├── train.py                              ← training entry point
├── evaluate.py                           ← evaluation entry point
├── visualize_ego_scenes.py               ← scene visualization
└── eval_top_confidence.py                ← top-confidence ADE/FDE evaluation
```

---

## 6. Preprocessing

Preprocessing reads nuScenes data and saves one pickle file per prediction sample. These pickles are what the training DataLoader reads — preprocessing only needs to be run once.

### 6.1 Run ego preprocessing

```bash
cd /path/to/PGP_ego

python preprocess.py \
  -c configs/preprocess_nuscenes_ego.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed
```

**Arguments:**

| Flag | Description |
|------|-------------|
| `-c` | Preprocessing config YAML |
| `-r` | nuScenes data root (contains `v1.0-trainval/` and `maps/`) |
| `-d` | Output directory for pickle files (created if absent) |

**What it produces:**

```
pgp_ego_preprocessed/
├── stats.pickle                        ← dataset statistics
├── ego_<sample_token>.pickle           ← one per training sample
├── ego_<sample_token>.pickle
└── ...  (1,833 files total for v1.0-trainval)
```

**Runtime:** ~3–5 minutes on CPU (58 total mini-batches across all 3 splits).

**Token generation logic** (defined in `NuScenesEgoGraphs._build_ego_token_list`):
- Uses the same nuScenes prediction challenge scenes as the agent model
- A valid window requires **4 keyframes of history** (2s) and **12 keyframes of future** (6s)
- Windows are spaced 12 keyframes apart (non-overlapping) to avoid data leakage
- Result: ~2 windows per 20-second scene, ~1,150 train tokens, ~373 val tokens

### 6.2 Pickle file contents

Each `ego_{sample_token}.pickle` contains a dict:

```python
{
    'inputs': {
        'instance_token': 'ego',
        'sample_token': '<hex>',
        'map_representation': {
            'lane_node_feats':   np.ndarray  # [N, P, 6]  lane polylines
            'lane_node_masks':   np.ndarray  # [N, P, 1]  validity mask
            's_next':            np.ndarray  # [N, K]     successor node indices
            'edge_type':         np.ndarray  # [N, K]     edge type flags
        },
        'surrounding_agent_representation': {...},  # nearby agents
        'target_agent_representation': np.ndarray,  # [t_h*2+1, 5] ego history
        'agent_node_masks':  np.ndarray,
        'init_node':         np.ndarray,   # starting lane node index
        'node_seq_gt':       np.ndarray,   # ground-truth node sequence
    },
    'ground_truth': {
        'traj':   np.ndarray,   # [12, 2]  future trajectory in ego frame
        'evf_gt': np.ndarray,   # ground-truth edge visit flags
    }
}
```

**Feature layout for `lane_node_feats` [N, P, 6]:**

| Channel | Meaning |
|---------|---------|
| 0 | x position in ego frame |
| 1 | y position in ego frame |
| 2 | yaw angle |
| 3 | is stop line (0 or 1) |
| 4 | is pedestrian crossing (0 or 1) |
| 5 | is boundary (0 or 1) |

**Feature layout for `target_agent_representation` [t_h*2+1, 5]:**

| Channel | Meaning |
|---------|---------|
| 0 | x position in ego frame |
| 1 | y position in ego frame |
| 2 | speed (m/s) |
| 3 | acceleration (m/s²) |
| 4 | yaw rate (rad/s) |

### 6.3 Agent preprocessing (original PGP, for reference)

```bash
python preprocess.py \
  -c configs/preprocess_nuscenes.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_preprocessed
```

Produces ~49,788 agent pickle files. Takes ~2 hours on CPU.

---

## 7. Training from Scratch

### 7.1 Smoke test first (recommended)

Always verify the pipeline with a quick 2-epoch run before committing to full training:

```bash
python train.py \
  -c configs/pgp_ego_smoketest.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -o /path/to/smoketest_output \
  -n 2
```

Expected output:

```
Epoch (1/2)
Training: [====================] 100 %, ETA: 0s, Metrics: { min_ade_5: 1.3x, ... }
Validating: [====================] 100 %, ETA: 0s, Metrics: { min_ade_5: 1.2x, ... }
Epoch (2/2)
...
```

If no errors appear and metrics decrease, the pipeline is working.

### 7.2 Full training

```bash
python train.py \
  -c configs/pgp_ego_gatx2_lvm_traversal.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -o /path/to/pgp_ego_output \
  -n 50
```

**Arguments:**

| Flag | Description |
|------|-------------|
| `-c` | Training config YAML |
| `-r` | nuScenes data root |
| `-d` | Preprocessed pickle directory |
| `-o` | Output directory (checkpoints + TensorBoard logs) |
| `-n` | Number of epochs |
| `-w` | (Optional) Path to checkpoint to resume from |
| `--just-weights` | (Optional) Load only model weights, reset optimizer/epoch |

**Output directory layout:**

```
pgp_ego_output/
├── checkpoints/
│   ├── best.tar          ← best val metric checkpoint (overwritten when improved)
│   ├── 0.tar             ← end-of-epoch checkpoints
│   ├── 1.tar
│   └── ...
└── tensorboard_logs/
    └── events.out.tfevents.*
```

**Monitor TensorBoard:**

```bash
tensorboard --logdir /path/to/pgp_ego_output/tensorboard_logs
```

**Expected training time (T4 GPU):** ~30s/epoch with ~1,150 train tokens and `batch_size=32` → ~25 minutes for 50 epochs.

**Training metrics reported:**

| Metric | Description |
|--------|-------------|
| `min_ade_5` | min-ADE over best 5 of K predictions |
| `min_ade_10` | min-ADE over best 10 of K predictions |
| `miss_rate_5` | fraction of samples where best-of-5 FDE > 2m |
| `miss_rate_10` | fraction of samples where best-of-10 FDE > 2m |
| `pi_bc` | policy imitation behavioral cloning loss |

---

## 8. Fine-tuning from a Checkpoint

### 8.1 Fine-tune from the agent model (recommended starting point)

The agent model checkpoint (`pgp_output/checkpoints/best.tar`) provides a strong initialization. Fine-tune using `--just-weights` so the optimizer and epoch counter reset to 0:

```bash
python train.py \
  -c configs/pgp_ego_gatx2_lvm_traversal.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -o /path/to/pgp_ego_output \
  -n 50 \
  -w /path/to/pgp_output/checkpoints/best.tar \
  --just-weights
```

> **Why `--just-weights`?** Without this flag, the trainer loads the epoch counter from the checkpoint (e.g., epoch 100 for the agent model), and training will run epochs 100→150 instead of 0→50. `--just-weights` discards optimizer state and resets the epoch counter so you get a clean fine-tuning run.

### 8.2 Resume an interrupted ego training run

If training was interrupted, resume without `--just-weights`:

```bash
python train.py \
  -c configs/pgp_ego_gatx2_lvm_traversal.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -o /path/to/pgp_ego_output \
  -n 50 \
  -w /path/to/pgp_ego_output/checkpoints/42.tar   # last saved epoch
```

This restores the optimizer state, scheduler, and epoch counter, resuming from where training stopped.

### 8.3 Checkpoint format

Checkpoints are `.tar` files saved by `torch.save`:

```python
{
    'epoch': int,                    # next epoch to run
    'model_state_dict': OrderedDict, # model weights
    'optimizer_state_dict': dict,    # AdamW state
    'scheduler_state_dict': dict,    # StepLR state
    'val_metric': float,             # validation metric at this epoch
    'min_val_metric': float,         # best validation metric seen so far
}
```

Load a checkpoint manually:

```python
import torch
ckpt = torch.load('best.tar', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
```

---

## 9. Evaluation

### 9.1 Standard metrics (minADE, minFDE, miss rate)

Uses `evaluate.py`, which runs on the **test split** (`val` in nuScenes terminology):

```bash
python evaluate.py \
  -c configs/pgp_ego_gatx2_lvm_traversal.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -o /path/to/eval_output \
  -w /path/to/pgp_ego_output/checkpoints/best.tar
```

Results are written to `eval_output/results/results.txt` and printed to stdout.

> Note: `evaluate.py` uses the `test_set_args` split (`val` split in nuScenes). Ensure the `val` split pickles are present in the preprocessed directory. The current preprocessing config generates data for all three splits (train, train_val, val) in one run.

### 9.2 Top-confidence ADE/FDE

Measures the ADE and FDE of the **single highest-confidence trajectory** (argmax of predicted probabilities). Uses the **validation split** (`train_val`):

```bash
python eval_top_confidence.py \
  -c configs/pgp_ego_gatx2_lvm_traversal.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -w /path/to/pgp_ego_output/checkpoints/best.tar \
  --batch_size 32 \
  --num_workers 4 \
  --num_samples 100
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `-c` | required | Config YAML |
| `-r` | required | nuScenes data root |
| `-d` | required | Preprocessed data directory |
| `-w` | required | Checkpoint path |
| `--batch_size` | 32 | Inference batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--num_samples` | 100 | Trajectory samples before clustering (100 is sufficient; 1000 is needed for best minADE quality but much slower) |

**Sample output:**

```
============================================================
FINAL RESULTS  (ego model, train_val split, highest-confidence traj)
============================================================
  Samples evaluated : 373
  Top-confidence ADE: 2.7260 m  (std=2.6803)
  Top-confidence FDE: 6.5163 m  (std=6.5079)
  minADE (best-of-K) : 0.7947 m  (reference)
  minFDE (best-of-K) : 1.4194 m  (reference)
============================================================

Top-confidence ADE percentile breakdown:
  p25: 0.8709 m
  p50: 1.8402 m
  p75: 3.5791 m
  p90: 6.2786 m
  p95: 8.1194 m
```

**How confidence scores work:** The LVM decoder clusters `num_samples` trajectories into `num_clusters` (default 10) modes via k-means. Each cluster's score is `1 / rank` where rank 1 = most-populated cluster. Scores are then normalized to sum to 1. Therefore `argmax(probs)` always selects the largest cluster (the "most common" trajectory mode the decoder samples).

---

## 10. Visualization

Generates 3-panel PNG images for specified nuScenes scenes: HD map, predicted trajectories, and ground truth.

```bash
python visualize_ego_scenes.py \
  -c configs/pgp_ego_gatx2_lvm_traversal.yml \
  -r /path/to/nuscenes_data \
  -d /path/to/pgp_ego_preprocessed \
  -w /path/to/pgp_ego_output/checkpoints/best.tar \
  -o /path/to/vis_output \
  -s 28 31 61
```

**Arguments:**

| Flag | Description |
|------|-------------|
| `-c` | Config YAML |
| `-r` | nuScenes data root |
| `-d` | Preprocessed data directory |
| `-w` | Checkpoint path |
| `-o` | Output directory for PNG files |
| `-s` | nuScenes scene numbers (space-separated integers → scene-0028, scene-0031, scene-0061) |

**How scene numbers work:**

Scene numbers map directly to nuScenes scene names. `--scenes 28 31 61` corresponds to scenes `scene-0028`, `scene-0031`, `scene-0061` by the format `f'scene-{num:04d}'`. The prediction frame is placed at **index 8** (4 seconds at 2 Hz) from the first sample of each scene.

**Output per scene:**

A PNG file named `ego_scene-XXXX.png` with 3 side-by-side panels on a dark background:

| Panel | Content |
|-------|---------|
| Left | HD map: lane polylines (gray = road, blue = pedestrian crossing, red = stop line) |
| Middle | K=10 sampled predicted trajectories (plasma colormap, best 5 + 5 random) |
| Right | Ground truth trajectory (green line, diamond at endpoint) |

The figure title shows scene name, prediction time, and ADE/FDE metrics.

**Map coordinate system:**

- Origin (cyan star) = ego vehicle at prediction time
- Cyan arrow = forward heading direction
- `local_y` (up in image) = forward
- `local_x` (right in image) = right lateral
- `MAP_EXTENT = [-50, 50, -20, 80]` = 100m wide × 100m long (20m behind, 80m ahead)

> If the scene's pickle is not in the preprocessed directory, the script computes data on-the-fly using the dataset class. This requires the nuScenes dataset and slightly more memory.

---

## 11. Config File Reference

### 11.1 Preprocessing config (`preprocess_nuscenes_ego.yml`)

```yaml
dataset: 'nuScenes'
version: 'v1.0-trainval'
agent_setting: 'single_agent'
input_representation: 'ego_graphs'   # maps to NuScenesEgoGraphs class

train_set_args:
  split: 'train'
  t_h: 2                # history window in seconds
  t_f: 6                # prediction horizon in seconds
  map_extent: [-50, 50, -20, 80]    # [x_min, x_max, y_min, y_max] in metres
  polyline_resolution: 1             # metres between polyline points
  polyline_length: 20                # max points per polyline segment
  traversal_horizon: 15              # max lane nodes in traversal sequence

val_set_args:
  split: 'train_val'
  ...

test_set_args:
  split: 'val'
  ...
```

The `input_representation: 'ego_graphs'` key is what causes `initialize_dataset` to instantiate `NuScenesEgoGraphs` instead of the regular agent graph class.

### 11.2 Training config (`pgp_ego_gatx2_lvm_traversal.yml`)

Key fields relevant to training behaviour:

```yaml
batch_size: 32         # reduce to 16 if VRAM is < 12 GB
num_workers: 4

aggregator_args:
  num_samples: 1000    # trajectory samples per forward pass; reduce to 100 to save VRAM

decoder_args:
  num_samples: 1000    # must match aggregator num_samples
  num_clusters: 10     # output modes (K)
  op_len: 12           # output timesteps = t_f * 2

optim_args:
  lr: 0.001
  scheduler_step: 10   # LR decay every N epochs
  scheduler_gamma: 0.5 # LR multiplied by this at each step

log_freq: 5            # print metrics every N mini-batches
                       # IMPORTANT: must be <= num_batches_per_epoch
                       # (for ego with ~1150 train tokens and batch_size=32,
                       #  num_batches ≈ 36 — keep log_freq <= 5)
```

### 11.3 `log_freq` caveat

`log_period = len(train_dataloader) // log_freq`. If `log_freq > len(train_dataloader)`, this equals 0 and causes a `ZeroDivisionError`. The trainer guards against this:

```python
self.log_period = max(1, len(self.tr_dl) // cfg['log_freq'])
```

But still set `log_freq` to a value ≤ the number of batches per epoch. For the ego dataset with ~1,150 train tokens and `batch_size=32`, that is ~36 batches/epoch — use `log_freq: 5` or lower.

---

## 12. Model Architecture

```
Input (per sample)
  ├── Target agent history  [t_h*2+1, 5]   (x, y, speed, accel, yaw_rate)
  ├── Lane node features    [N, P, 6]       (x, y, yaw, stop, ped_cross, boundary)
  ├── Lane node masks       [N, P, 1]
  ├── Neighbor agents       [A, t_h*2+1, 5]
  ├── init_node             [1]             (starting lane node index)
  └── node_seq_gt           [traversal_horizon]  (GT traversal sequence, training only)
         │
         ▼
┌─────────────────────────────────────┐
│  PGP Encoder (GAT × 2)             │
│  - Target agent MLP + GRU          │
│  - Lane node GRU per polyline      │
│  - Neighbour agent GRU             │
│  - Graph Attention Network (×2)    │
│  Output: node encodings [N, 32]    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  PGP Aggregator (lane-graph walk)  │
│  - Samples K=1000 traversals       │
│  - Each traversal: probabilistic   │
│    random walk over lane graph     │
│  - Aggregates node encodings along │
│    each traversal path             │
│  Output: [B, K, 160] encodings     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  LVM Decoder (Latent Variable)     │
│  - Samples K Gaussian latent vars  │
│  - MLP: 160 + lv_dim → 128 → 24   │
│  - Reshape to [K, 12, 2] trajs     │
│  - k-means cluster to 10 modes     │
│  Output: traj [B, 10, 12, 2]       │
│          probs [B, 10]             │
└─────────────────────────────────────┘
```

**Parameter count:** ~149,000 parameters total.

**Coordinate system:** All positions are in the **ego-centric frame** at prediction time. `local_y` is the forward direction (positive = ahead), `local_x` is the lateral direction (positive = right). The rotation used in `global_to_local` is `(π/2 − yaw)`, where `yaw` is from `quaternion_yaw` applied to the LIDAR_TOP sensor's ego_pose rotation.

---

## 13. Troubleshooting

### `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`

A checkpoint was saved on a GPU machine and is being loaded on CPU. The fix is already applied in `train_eval/trainer.py`:

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
```

If you load checkpoints manually, always pass `map_location`:

```python
ckpt = torch.load('best.tar', map_location=torch.device('cpu'))
```

### `ZeroDivisionError: integer modulo by zero` during training

`log_freq` is larger than the number of batches per epoch. Reduce `log_freq` in your config to ≤ number of batches. For the ego dataset: set `log_freq: 5` or lower. The guard in `trainer.py` (`max(1, ...)`) prevents a crash but the printed metrics may be misleading if `log_freq` is too large.

### `AttributeError: 'NuScenesEgoGraphs' object has no attribute 'max_nodes'`

This appears when computing data on-the-fly (not loading from pickles). The `max_nodes`, `max_vehicles`, `max_pedestrians`, and `max_nbr_nodes` stats are only loaded in `extract_data` mode, not `load_data` mode. The `visualize_ego_scenes.py` script handles this with `ensure_compute_stats(ds)`. If you write your own inference script:

```python
stats = ds.load_stats()
ds.max_nodes = stats['num_lane_nodes']
ds.max_vehicles = stats['num_vehicles']
ds.max_pedestrians = stats['num_pedestrians']
ds.max_nbr_nodes = stats['max_nbr_nodes']
```

### `KeyError: 's_next'` inside the aggregator

The model's aggregator adds `s_next` to node encodings only when `'init_node' in inputs`. If `init_node` is missing, the aggregator skips this step and the forward pass raises a `KeyError`. This happens when calling `get_inputs()` alone without the `extract_data()` extras. The fix:

```python
node_seq_gt, evf_gt = ds.get_visited_edges(0, inputs['map_representation'])
init_node = ds.get_initial_node(inputs['map_representation'])
inputs['init_node'] = init_node
inputs['node_seq_gt'] = node_seq_gt
ground_truth['evf_gt'] = evf_gt
```

### `ray.exceptions.OutOfMemoryError` during inference

nuScenes `v1.0-trainval` metadata (~13.4 GB) leaves little RAM for Ray worker forks. Fix (already in `visualize_ego_scenes.py` and `eval_top_confidence.py`):

```python
import os
os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')
# Must be set BEFORE importing ray or any PGP module that imports ray
```

Also reduce `num_samples` to 100 for visualization (the quality difference is minimal):

```python
model.aggregator.num_samples = 100
model.decoder.num_samples = 100
```

### `TypeError: can't convert cuda:0 device type tensor to numpy`

Tensors sent to GPU via `u.send_to_device()` must be moved back before calling `.numpy()`:

```python
array = tensor.detach().cpu().numpy()
```

### Preprocessing `FileNotFoundError` for stale pickles

If you change the token list (e.g., after fixing the history/future window logic), the old pickle files will no longer match the new token names. Clear the data directory and re-run preprocessing:

```bash
rm -rf /path/to/pgp_ego_preprocessed/
python preprocess.py -c configs/preprocess_nuscenes_ego.yml -r ... -d /path/to/pgp_ego_preprocessed
```

### Scene number confusion

`visualize_ego_scenes.py` accepts nuScenes scene numbers (integers), which map to scene names as `f'scene-{num:04d}'`. So `--scenes 28 31 61` corresponds to `scene-0028`, `scene-0031`, `scene-0061`. This is **not** the index into the val split token list — it is the nuScenes scene number as assigned in the dataset.
