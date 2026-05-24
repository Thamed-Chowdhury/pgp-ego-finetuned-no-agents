# pgp-ego-finetuned-no-agents

**Language-conditioned PGP ego trajectory predictor for the doScenes Ablation / No-Agents track.**

This repository extends the confidence-trained PGP-DSVT checkpoint
([PGP-DSVT_ego_confidence_trained](https://github.com/Thamed-Chowdhury/PGP-DSVT_ego_confidence_trained))
with a **text-conditioning pathway**: doScenes driver instructions are encoded
via Google's `gemini-embedding-001` (768-d) and residual-injected into PGP's
per-sample aggregated encoding inside a new `LVMRankedText` decoder. Both the
trajectory MLP and the learnable confidence head therefore see the
instruction. The full pipeline runs **without any surrounding-agent input at
test time** — only the ego history, the HD map, and the instruction
embedding — making it directly applicable to the doScenes no-agents /
ablation evaluation protocol on the nuScenes v1.0-test split.

> **Headline:** On the 150-scene doScenes-style test protocol with no agents at
> inference, the text-conditioned model achieves **ADE\@6s = 2.6526 m** vs the
> stage-2 non-text confidence-trained model's **2.6862 m** (Δ_ADE = +0.034 m
> in favour of language). The same checkpoint also produces the
> *without-language* ablation number when fed a zero text embedding (see §3.2).

---

## 1. What this repo adds

Building on top of the prior confidence-trained PGP-DSVT submission:

- **`LVMRankedText` decoder** ([PGP_ego/models/decoders/lvm_ranked_text.py](PGP_ego/models/decoders/lvm_ranked_text.py))
  — same body as `LVMRanked` (1000 latent samples → decode → K-means → K=10
  centroids + learnable confidence head), with one new layer:
  `text_proj: Linear(768 → encoding_size=160)`. The projected text embedding
  is **residual-added** to the per-sample `agg_encoding` so that both
  trajectory generation (the LVM op_traj) and the confidence head see it.
  `text_proj` is **zero-initialised** so the model starts bit-identical to
  `LVMRanked` when warm-started from a stage-2 checkpoint.
- **`NuScenesEgoGraphsText` dataset wrapper** ([PGP_ego/datasets/nuScenes/nuScenes_ego_graphs_text.py](PGP_ego/datasets/nuScenes/nuScenes_ego_graphs_text.py))
  — subclass of `NuScenesEgoGraphs` that loads a pickle of cached
  `gemini-embedding-001` instruction embeddings and injects
  `inputs['text_embedding']` for every ego window. Scenes with no doScenes
  annotation get the zero embedding (no-op under residual addition).
- **Gemini embedding builder** ([PGP_ego/embed_doscenes_gemini.py](PGP_ego/embed_doscenes_gemini.py))
  — pools all 12 doScenes annotator CSVs by scene number with priority
  `d → s → sd → ds → other`, embeds every unique instruction via
  `gemini-embedding-001` at output_dim=768, caches to a single pickle.
- **Text pass-through** in [PGP_ego/models/encoders/pgp_encoder.py](PGP_ego/models/encoders/pgp_encoder.py)
  and [PGP_ego/models/aggregators/pgp.py](PGP_ego/models/aggregators/pgp.py)
  — these existing modules now forward `inputs['text_embedding']` through
  their output dicts so the decoder can consume it. No-op when absent.
- **Stage-3 training config** [PGP_ego/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml](PGP_ego/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml)
  — fine-tunes from `lvm_ranked` stage-2 best.tar (loaded with
  `--just-weights`, non-strict), 50 epochs, lr 2e-4, batch 16.
- **Test eval drivers**:
  - [PGP_ego/run_doscenes_test_text_conditioned.py](PGP_ego/run_doscenes_test_text_conditioned.py) — with-language (full instruction embedding).
  - [PGP_ego/run_doscenes_test_text_conditioned_no_language.py](PGP_ego/run_doscenes_test_text_conditioned_no_language.py) — ablation: same checkpoint, zero text embedding.
- **Stage-3 best checkpoint** [checkpoints/stage3_best.tar](checkpoints/stage3_best.tar)
  — epoch 1, val_metric=1.156 (3.8 MB).
- **Cached embeddings** [doscenes_embeddings/doscenes_gemini_embeddings.pkl](doscenes_embeddings/doscenes_gemini_embeddings.pkl)
  — 620 unique doScenes instructions × 768-d (~2 MB).

The DSVT injection pipeline, the encoder/aggregator code, the dataset code,
and the rank-aware loss / confidence head from the prior submission are all
reused unchanged.

---

## 2. doScenes Ablation track interpretation

The doScenes Ablation track requires both **with-language** and
**without-language** numbers under an identical protocol. The same stage-3
checkpoint covers both:

| Setting | Model input | Eval driver |
|---|---|---|
| With language | real `gemini-embedding-001` embedding of the scene's doScenes instruction (zero vector for 23 unannotated scenes) | [run_doscenes_test_text_conditioned.py](PGP_ego/run_doscenes_test_text_conditioned.py) |
| Without language (ablation) | zero text embedding for every scene | [run_doscenes_test_text_conditioned_no_language.py](PGP_ego/run_doscenes_test_text_conditioned_no_language.py) |

This gives a clean Δ_ADE while keeping the model checkpoint, the K-means
clustering scheme, and every other inference variable identical.

---

## 3. Results on the 150-scene v1.0-test protocol

All numbers are top-1 confidence (argmax over the K=10 cluster scores),
evaluated by [doScenes_repo/metrics.py](https://github.com/rossgreer/doScenes/blob/main/metrics.py)
on the 150-scene doScenes-style test split (one window per scene at
`samples[4]`).

### 3.1 With-language (full instruction)

| Metric | Value |
|--------|------:|
| ade_2s | 0.6034 |
| ade_4s | 1.5008 |
| **ade_6s** | **2.6526** |
| fde | 6.0935 |
| miss_rate | 0.7800 |
| offroad | 0.0006 |

### 3.2 Comparison vs the non-text confidence-trained model

**Apples-to-apples ablation — same stage-3 checkpoint, only the text input toggled:**

| Setting | ade_2s | ade_4s | **ade_6s** | fde | miss_rate | offroad |
|---|---:|---:|---:|---:|---:|---:|
| **With language** (real gemini-embedding-001) | 0.6034 | 1.5008 | **2.6526** | 6.0935 | 0.7800 | 0.0006 |
| Without language (zero text embedding)        | 0.6213 | 1.5320 | **2.7119** | 6.2239 | 0.7667 | 0.0012 |
| **Δ_ADE (without − with)**                    | +0.018 | +0.031 | **+0.0593** | +0.130 | −0.013 | +0.0006 |

**Reading.** Holding the model checkpoint fixed and only flipping the text
input from a real instruction embedding to a zero vector raises ADE\@6s by
**0.059 m (~2.2%)**. The language input contributes consistently across
short and long horizons (ade_2s +0.018, ade_4s +0.031, ade_6s +0.059, fde
+0.130). Miss rate moves the other way by 0.013 — within noise on 150
scenes.

Comparing across checkpoints (less clean — different K-means scheme):
the stage-2 non-text confidence-trained model scored ade_6s = 2.6862 m on
the same test pickles. Stage-3 with-text (2.6526 m) is 0.034 m better, and
stage-3 without-text (2.7119 m) is 0.026 m worse — both within K-means
stochasticity.

Note: the stage-3 best.tar fell on **epoch 1** (val_metric=1.156). After
warm-starting from stage-2 best with the zero-initialised `text_proj`, the
model improved on its first val pass and never beat that pass — subsequent
epochs trained `text_proj` further but did not lower the val metric. The
practical implication: the small per-scene effect of text on the no-agents
variant is mostly carried by a barely-trained `text_proj`. A longer
fine-tune with an explicit language-aware curriculum (e.g. up-weighting
direction-explicit instructions) is the natural next step.

### 3.3 Stage-3 training curve

| Stage 3 epoch | val min_ade_5 | val min_ade_10 | val top1_ade |
|---:|---:|---:|---:|
| 1 (best) | **1.156** | 0.70 | 2.40 |
| 17 | 1.17 | 0.71 | 2.35 |
| 31 | 1.18 | 0.69 | 2.32 |
| 49 | 1.20 | 0.70 | 2.29 |
| 50 (final) | 1.20 | 0.70 | 2.35 |

`top1_ade` continued to fall through training (2.40 → ~2.30), but the
selection metric `min_ade_5` plateaued, so the best.tar checkpoint was
overwritten only once.

---

## 4. Submission CSV format

Per the [doScenes challenge spec](https://mi3-lab.github.io/doScenes_challenge#leaderboard),
all submission CSVs use this header:

    sample_token, instruction, x1, y1, x2, y2, ..., x12, y12

The `instruction` column is **always present**, even for the no-language
baseline. For the with-language submission it contains the doScenes
instruction string the pipeline fed to the model (empty for the 23 scenes
with no annotation). For the no-language ablation it is empty for every
row.

Both eval drivers in this repo
([run_doscenes_test_text_conditioned.py](PGP_ego/run_doscenes_test_text_conditioned.py)
and
[run_doscenes_test_text_conditioned_no_language.py](PGP_ego/run_doscenes_test_text_conditioned_no_language.py))
already emit submissions in this format. If you have a submission CSV from
an older run without the column, you can fix it in place with
[PGP_ego/add_instruction_column.py](PGP_ego/add_instruction_column.py):

```bash
python3 PGP_ego/add_instruction_column.py \
    --test_root nuscenes_data/v1-test \
    --ann_dir   /path/to/doScenes_repo/Annotations \
    --in_csv    old_submission.csv \
    --out_csv   new_submission.csv \
    --used_language     # omit for the no-language ablation
```

## 5. How to use

### 5.1 When is DSVT actually needed?

**Inference for the no-agents track does NOT use DSVT.** The shipped
checkpoint reads `pgp_ego_test_preprocessed/` whose
`surrounding_agent_representation` block is an all-zero tensor; the eval
drivers ([run_doscenes_test_text_conditioned.py](PGP_ego/run_doscenes_test_text_conditioned.py)
and [run_doscenes_test_text_conditioned_no_language.py](PGP_ego/run_doscenes_test_text_conditioned_no_language.py))
import no DSVT module and make no DSVT call. Concretely, a no-agents
v1.0-test pickle that this submission consumes:

```
surrounding_agent_representation.vehicles    : shape (78, 5, 5), all zeros
surrounding_agent_representation.pedestrians : shape (74, 5, 5), all zeros
```

The model sees only ego history + HD map + the
[gemini-embedding-001](https://ai.google.dev/gemini-api/docs/embeddings)
text embedding.

**Training did use DSVT**, transitively, through the warm-start chain:
the stage-1 PGP non-ego pretrain and the stage-2 PGP-ego fine-tune both
consumed `pgp_*_preprocessed_dsvt/` pickles (DSVT detections injected as
surrounding agents). Stage 3 in this repo warm-starts from stage-2 best.tar
and continues fine-tuning on the same DSVT-injected ego pickles. So the
weights themselves were shaped by DSVT exposure during training, even though
no DSVT data ever reaches the model at inference for the no-agents track.

### 5.2 Prerequisites

What you need depends on what you want to do:

**(a) Reproduce the 2.6526 m result with the shipped stage-3 checkpoint
(no DSVT setup required):**

- nuScenes v1.0-test metadata + maps + LiDAR sweeps at `nuscenes_data/v1-test/`.
  *(LiDAR is only needed for the dataset class's preprocessing pass; it is
  never read at inference.)*
- A trainval `stats.pickle` (upper-bound padding sizes; one is shipped with
  any prior preprocessing run).
- doScenes_repo for the official `compute_ego_metrics`.
- A Gemini API key with access to `gemini-embedding-001`. Put one or more
  keys, one per line, in `PGP_ego/Gemini_keys.txt` (this file is
  **gitignored** in this repo).
- This repo's `checkpoints/stage3_best.tar` and `doscenes_embeddings/doscenes_gemini_embeddings.pkl`.
- **No DSVT install. No DSVT-injected pickles. No DSVT detector checkpoint.**

**(b) Re-train stage 3 from a stage-2 LVMRanked checkpoint:**

- All of (a), plus
- A `lvm_ranked` stage-2 best.tar to warm-start from (the published one is
  at [PGP-DSVT_ego_confidence_trained](https://github.com/Thamed-Chowdhury/PGP-DSVT_ego_confidence_trained/blob/main/checkpoints/stage2_best.tar)).
- `pgp_ego_preprocessed_dsvt/` (1,832 ego windows with DSVT detections
  injected as surrounding agents — produced by the prior submission's pipeline;
  see [PGP-DSVT-ego_finetuned](https://github.com/Thamed-Chowdhury/PGP-DSVT-ego_finetuned)
  for the DSVT injection scripts).

**(c) Reproduce the entire pipeline from scratch (DSVT-injection upwards):**
- All of (b), plus DSVT-pillar checkpoint, nuScenes v1.0-trainval, and the
  injection / training infrastructure from the two sister repos.

For most reviewers, **(a) is the only path that needs to be runnable**, and
it has no DSVT dependency at all.

### 5.3 Python environment

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.8.0+cu128 |
| nuscenes-devkit | 1.2.0 |
| ray | 2.55.0 |
| google-genai | 2.4.0 |
| pyquaternion | any recent |
| scikit-learn | any recent |

### 5.4 Embed doScenes instructions (once)

```bash
python3 PGP_ego/embed_doscenes_gemini.py \
    --ann_dir   /path/to/doScenes_repo/Annotations \
    --keys      PGP_ego/Gemini_keys.txt \
    --out       doscenes_embeddings/doscenes_gemini_embeddings.pkl
```

Takes ~30 s for ~620 unique instructions (with key rotation). A copy of the
output is already shipped at
`doscenes_embeddings/doscenes_gemini_embeddings.pkl`.

### 5.5 Train stage 3 (optional — checkpoint is shipped)

```bash
python3 PGP_ego/train.py \
    -c PGP_ego/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml \
    -r nuscenes_data -d pgp_ego_preprocessed_dsvt \
    -o pgp_ego_output_text_stage3 -n 50 \
    -w /path/to/lvm_ranked_stage2_best.tar \
    --just-weights
```

~30 s/epoch × 50 epochs ≈ 25 min on a single L4.

### 5.6 Eval — with language

```bash
python3 PGP_ego/run_doscenes_test_text_conditioned.py \
    --config         PGP_ego/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml \
    --test_root      nuscenes_data/v1-test \
    --trainval_stats pgp_ego_preprocessed/stats.pickle \
    --test_preproc   pgp_ego_test_preprocessed \
    --checkpoint     checkpoints/stage3_best.tar \
    --doscenes_repo  /path/to/doScenes_repo \
    --text_emb_pkl   doscenes_embeddings/doscenes_gemini_embeddings.pkl \
    --out_dir        out_baseline_withlang \
    --skip_preprocess --batch_size 4
```

Writes `out_baseline_withlang/submission.csv` and
`out_baseline_withlang/self_eval_metrics.json`. Expected ADE\@6s ≈ 2.65 m.

### 5.7 Eval — without language (ablation)

```bash
python3 PGP_ego/run_doscenes_test_text_conditioned_no_language.py \
    --config         PGP_ego/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml \
    --test_root      nuscenes_data/v1-test \
    --trainval_stats pgp_ego_preprocessed/stats.pickle \
    --test_preproc   pgp_ego_test_preprocessed \
    --checkpoint     checkpoints/stage3_best.tar \
    --doscenes_repo  /path/to/doScenes_repo \
    --out_dir        out_baseline_nolang
```

---

## 6. Architecture diagram

```
 Inputs (Ego history, HD map, ZERO surrounding agents at test)
                       │
                       ▼
              ┌──────────────────────┐
              │  PGP encoder (GAT×2) │
              │  + PGP aggregator    │
              └──────────────────────┘
                       │ agg_encoding (B, num_samples, 160)
                       ▼
            ┌──────────────────────────┐
            │   text_proj (768→160)    │◀── gemini-embedding-001
            │   residual add           │     of doScenes instruction
            └──────────────────────────┘     (zero vec if no annotation)
                       │ (modified agg_encoding)
                       ▼
              ┌──────────────────────────┐
              │  LVM body + K-means      │
              │  → K=10 centroids        │
              │  → confidence head       │
              │    (also sees text       │
              │    via global ctx)       │
              └──────────────────────────┘
                       │ K trajectories + log_probs
                       ▼
                argmax → top-1 trajectory
                       │
                       ▼
               submission CSV
```

Both the trajectory generation and the confidence head consume the text
signal through the **single** `text_proj` residual addition — language
shapes which mode is decoded *and* which mode is ranked highest.

---

## 7. Repository layout

```
.
├── README.md
├── .gitignore
├── PGP_ego/                     (PGP code with text-conditioning hooks)
│   ├── models/decoders/lvm_ranked_text.py        ← new
│   ├── models/decoders/lvm_ranked.py
│   ├── models/encoders/pgp_encoder.py            ← edited (text passthrough)
│   ├── models/aggregators/pgp.py                 ← edited (text passthrough)
│   ├── datasets/nuScenes/nuScenes_ego_graphs_text.py  ← new
│   ├── metrics/ranking_xent.py
│   ├── metrics/top1_ade.py
│   ├── train_eval/initialization.py              ← registrations
│   ├── train_eval/trainer.py                     ← non-strict --just-weights
│   ├── configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml   ← new
│   ├── embed_doscenes_gemini.py                  ← new
│   ├── run_doscenes_test_text_conditioned.py     ← new
│   ├── run_doscenes_test_text_conditioned_no_language.py  ← new
│   └── ... (PGP infrastructure unchanged)
├── checkpoints/
│   └── stage3_best.tar           ← LVMRankedText, epoch 1, val_metric 1.156
├── doscenes_embeddings/
│   └── doscenes_gemini_embeddings.pkl  ← gemini-embedding-001 cache (620 instructions × 768)
├── scripts/
│   └── run_text_conditioned_test_evals.sh
├── test_results/                 (per-variant submission.csv + self_eval_metrics.json)
│   ├── baseline/                  ← no agents at inference (the 2.6526 m result)
│   ├── dsvt_motion/
│   └── dsvt_presence/
└── Exp Deterministic PGP/        ← deterministic reproducer (seed=21 → ADE@6s = 2.6112 m)
    ├── README.md
    ├── run_final_deterministic.sh
    ├── run_deterministic_inference.py
    ├── seed_sweep_inproc.py
    ├── checkpoints/stage3_best.tar
    ├── configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml
    ├── data/                      (stats.pickle + gemini-embedding-001 cache)
    └── results/
        ├── HEADLINE.json          ← machine-readable headline numbers
        └── FINAL_seed_21/         ← canonical bit-identical submission
```

---

## 8. Citation

```bibtex
@misc{chowdhury2026pgpegonoagents,
  title        = {{pgp-ego-finetuned-no-agents}: Text-conditioned PGP ego
                  trajectory predictor for the doScenes Ablation track},
  author       = {Chowdhury, Md Thamed Bin Zaman and Shahbaz, Muhammad and Agarwal, Shaurya},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/Thamed-Chowdhury/pgp-ego-finetuned-no-agents}
}

@inproceedings{roy2025doscenes,
  title     = {{doScenes}: An autonomous driving dataset with natural language
               instruction for human interaction and vision-language navigation},
  author    = {Roy, Parthib and Perisetla, Srinivas and Shriram, Shashank and
               Krishnaswamy, Harsha and Keskar, Aaditya and Greer, Ross},
  booktitle = {Proceedings of the 2025 IEEE 28th International Conference on
               Intelligent Transportation Systems (ITSC)},
  pages     = {1651--1658}, year = {2025}, publisher = {IEEE},
  url       = {https://arxiv.org/abs/2412.05893}
}

@article{deo2021pgp,
  title   = {Multimodal trajectory prediction conditioned on lane-graph traversals},
  author  = {Deo, Nachiket and Wolff, Eric M. and Beijbom, Oscar},
  journal = {arXiv preprint arXiv:2106.15004}, year = {2021},
  url     = {https://arxiv.org/abs/2106.15004}
}

@article{caesar2020nuscenes,
  title   = {{nuScenes}: A multimodal dataset for autonomous driving},
  author  = {Caesar, Holger and others}, journal = {arXiv:1903.11027}, year = {2020}
}
```

Sister repos:
- [PGP-DSVT_ego_confidence_trained](https://github.com/Thamed-Chowdhury/PGP-DSVT_ego_confidence_trained) — produces the warm-start checkpoint and the rank-aware confidence head.
- [PGP-DSVT-ego_finetuned](https://github.com/Thamed-Chowdhury/PGP-DSVT-ego_finetuned) — DSVT injection + two-stage retraining baseline.
- Upstream PGP: <https://github.com/nachiket92/PGP>
- Upstream DSVT: <https://github.com/Haiyang-W/DSVT>
- doScenes challenge: <https://github.com/rossgreer/doScenes>

---

## 9. License

MIT (inherits the upstream PGP license — see `PGP_ego/LICENSE`).

---

## 10. Authors

- **Md Thamed Bin Zaman Chowdhury** — PhD Student, CECE, UCF
- **Muhammad Shahbaz** — Post-Doctoral Scholar, UCF
- **Shaurya Agarwal** — Associate Professor, UCF
