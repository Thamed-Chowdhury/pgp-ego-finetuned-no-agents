# Data for the Deterministic PGP experiment

This folder ships two small artefacts that the deterministic reproducer needs
**at inference time**:

| file                              | size  | what it is                                                                                  |
|-----------------------------------|-------|---------------------------------------------------------------------------------------------|
| `stats.pickle`                    | 4 KB  | Trainval upper-bound padding sizes (output of PGP_ego's `preprocess.py compute_stats`)      |
| `doscenes_gemini_embeddings.pkl`  | 1.9 MB| `gemini-embedding-001` sentence embeddings for the 620 unique doScenes instructions (768-d) |

It also needs a third artefact that is **too large to commit** (~ 87 MB):

| missing                | what to do                                                                                       |
|------------------------|--------------------------------------------------------------------------------------------------|
| `test_preproc/`        | Symlink (or copy) this folder to a local directory of preprocessed v1.0-test ego pickles.        |

## Recreating `test_preproc/`

If you already have a preprocessed test directory from the original LangHistory
pipeline (e.g. `pgp_ego_test_preprocessed/` produced by
`run_doscenes_test_text_conditioned.py` without `--skip_preprocess`), just
symlink it:

```bash
ln -sfn /absolute/path/to/pgp_ego_test_preprocessed \
       "Exp Deterministic PGP/data/test_preproc"
```

If you don't have it yet, the original LangHistory reproducer in the parent
repo will create it for you on first run. After that, the symlink command
above is all you need to wire it into this experiment.

The reproducer scripts (`run_final_deterministic.sh`, `run_single_seed.sh`)
read `data/test_preproc` directly, so once the symlink is in place the
deterministic pipeline is fully self-contained.
