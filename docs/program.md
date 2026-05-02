# Jedi — Autonomous Research

This is an autonomous ML research setup for Jedi: BraTS2023 cross-modality contrast conversion using a le-wm-aligned two-stage JEPA pipeline.

## Architecture

Two scopes of code, inspired by karpathy/autoresearch:

- **prepare.py scope** (read-only): data loading, fixed configs, evaluation metrics, callbacks, tests, utilities
- **train.py scope** (modifiable): model architectures, training logic, losses, optimizers, training entry points

Jedi has two training stages:

1. **Stage 1 encoder-side latent alignment**: `src_volume` and `tgt_volume` are encoded by a shared 3D ViT stack. The predictor maps source latents toward target latents.
2. **Stage 2 decoder training**: the encoder-side stack is loaded from a Stage 1 checkpoint and frozen. Only the decoder is trained from predicted target latents to target volumes.

The default autoresearch loop optimizes **Stage 2 `val/loss`** using the existing Stage 1 checkpoint.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, e.g. `may03`. The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files**: Read these for full context:
   - Project docs: `README.md`, `CLAUDE.md`, `pyproject.toml`
   - **prepare.py scope** (read-only, do NOT modify):
     - `src/jedi/data/` — BraTS case discovery, modality pairing, MONAI transforms, persistent cache, dataloader construction
     - `src/jedi/config/` — Hydra config interface for data, model, trainer, and stage entry points
     - `src/jedi/training/callbacks.py` — fixed metric/logging callbacks for losses, latent metrics, reconstruction metrics, and dynamics
     - `src/jedi/training/logging.py` — resolved-config and WandB config logging infrastructure
     - `src/jedi/training/trainer_config.py` — Lightning Trainer config adapter
     - `src/jedi/utils.py` — checkpoint-loading utility used by Stage 2
     - `src/jedi/infer.py` — inference/evaluation-style CLI and checkpoint loading
     - `tests/` — fixed behavioral and regression tests
   - **train.py scope** (modifiable):
     - `src/jedi/models/` — 3D ViT, transformer blocks, predictor, JEPA wrapper, decoder architectures, regularizers, wavelet loss
     - `src/jedi/training/encoder_module.py` — Stage 1 LightningModule and latent alignment objective
     - `src/jedi/training/decoder_module.py` — Stage 2 LightningModule, frozen encoder-side path, PCGrad, L1 + wavelet objective
     - `src/jedi/training/optim.py` — AdamW parameter groups and warmup/cosine scheduler construction
     - `src/jedi/training/schedule.py` — total-step estimation feeding scheduler behavior
     - `src/jedi/train_encoder.py` — Stage 1 training composition and entry point
     - `src/jedi/train_decoder.py` — Stage 2 training composition and entry point
4. **Verify data/environment**:
   ```bash
   test -x .venv/bin/python
   PYTHONPATH=src .venv/bin/python -c "import jedi, torch, lightning, monai; print(jedi.__version__)"
   test -d "$BRATS_DATA_DIR"
   test -f "$ENCODER_CHECKPOINT"
   find "$BRATS_DATA_DIR" -maxdepth 2 -name "*.nii.gz" | head
   ```
   The BraTS data directory must contain case directories with `t1n`, `t1c`, `t2w`, and `t2f` NIfTI files. `ENCODER_CHECKPOINT` must point at a Stage 1 checkpoint compatible with the current encoder-side architecture.
5. **Initialize results.tsv**: Create `results.tsv` with the header row:
   ```
   commit	val/loss	memory_gb	status	description
   ```
6. **Initialize memory index**: Ensure the memory directory exists at `~/.claude/projects/<project-path>/memory/` and add an entry to `MEMORY.md` indexing the autoresearch run:
   ```
   - [Autoresearch <tag>](exp_<tag>_overview.md) — experiment run on 2026-05-03, optimizing Stage 2 val/loss
   ```
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs via:
```bash
RUN_TAG=baseline \
BRATS_DATA_DIR=/path/to/BraTS2023 \
ENCODER_CHECKPOINT=/path/to/stage1.ckpt \
PYTHONPATH=src .venv/bin/python -m jedi.train_decoder \
  data.data_dir="$BRATS_DATA_DIR" \
  decoder_model.encoder_checkpoint="$ENCODER_CHECKPOINT" \
  wandb.enabled=false \
  trainer.default_root_dir="autoresearch_runs/${RUN_TAG}" \
  hydra.run.dir="autoresearch_runs/${RUN_TAG}/hydra"
```

Training runs for the configured Stage 2 budget: `trainer.max_epochs=100`, `trainer.accumulate_grad_batches=4`, validation once per epoch, and checkpointing on the lowest `val/loss`.

For shorter exploratory smoke runs only, explicitly override the budget in the command, for example `trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=2`. Do not compare smoke-run metrics against full-run metrics.

**What you CAN do:**
- Modify files in train.py scope:
  - model architectures: `src/jedi/models/*.py`
  - training modules: `src/jedi/training/encoder_module.py`, `src/jedi/training/decoder_module.py`
  - loss and regularization code: `src/jedi/models/regularizers.py`, `src/jedi/models/wavelet_loss.py`
  - optimizer and scheduler construction: `src/jedi/training/optim.py`, `src/jedi/training/schedule.py`
  - training entry point composition: `src/jedi/train_encoder.py`, `src/jedi/train_decoder.py`
- Prefer Stage 2 decoder-side experiments by default, because the command loads a fixed Stage 1 checkpoint.
- If you change encoder-side architecture or latent dimensions, you must first run a matching Stage 1 training run and use its new checkpoint for Stage 2; otherwise Stage 2 checkpoint loading is not comparable.

**What you CANNOT do:**
- Modify files in prepare.py scope. They are read-only.
- Install new packages. Use only existing dependencies.
- Modify data loading, data transforms, fixed config files, callbacks, metric implementations, tests, or inference/evaluation harnesses.
- Change the BraTS train/validation modality mapping or validation transforms to make results easier.
- Reintroduce Stage 2 loss weight hyperparameters unless the human explicitly requests it.

**The goal: get the lowest `val/loss`.**
`val/loss` is Stage 2 validation reconstruction loss from `DecoderTrainingModule.validation_step`: `(l1_loss + wavelet_loss) / 2`. Lower is better. It is also the checkpoint monitor in `src/jedi/train_decoder.py`.

**Multi-metric discipline**: This project tracks multiple metrics. The primary target is `val/loss`, but **when a metric becomes the optimization target, it loses objectivity**. A change that improves `val/loss` while degrading secondary metrics is suspect. Always cross-reference:
- `val/l1_loss`: should decrease or remain stable when `val/loss` improves.
- `val/wavelet_loss`: should decrease or remain stable; large regressions mean the primary loss may be hiding frequency-domain artifacts.
- `val/mse`, `val/mae`, `val/rmse`, `val/nrmse`: should generally decrease with genuine reconstruction improvement.
- `val/psnr`: should increase when MSE improves.
- `val/ssim`: should increase or remain stable; a falling SSIM flags perceptual/structural degradation.
- `dynamics/grad_norm` and `dynamics/grad_to_param_norm`: should stay finite and not spike persistently.
- Peak memory: may increase for meaningful gains, but large jumps need a clear metric improvement.
Only declare an improvement genuine when the primary metric improves AND secondary metrics are stable or improving. If primary improves but a critical secondary degrades significantly, flag it for human review rather than auto-keeping.

**Memory** is a soft constraint. Some increase is acceptable for meaningful gains.

**Simplicity criterion**: All else being equal, simpler is better. For this project, treat `0.001` absolute `val/loss` as the minimum meaningful improvement unless repeated runs show a different run-to-run variance. A `0.001` `val/loss` improvement that adds `20` lines of hacky code is probably not worth it. A `0.001` `val/loss` improvement from deleting code is definitely worth keeping. An improvement of ~0 but much simpler code is also worth keeping. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Always establish the baseline first — run training as-is with no modifications.

## Output format

This project does not implement a custom final `print()` summary. Lightning writes metrics through `CSVLogger` under `logs/decoder_stage/version_*/metrics.csv`, and the command log contains Lightning progress/checkpoint output. After a completed Stage 2 run, extract and print a concrete summary like this:
```
primary_metric: 0.156234
val_loss: 0.156234
val_l1_loss: 0.128901
val_wavelet_loss: 0.183567
val_mse: 0.041230
val_mae: 0.128901
val_rmse: 0.203052
val_nrmse: 0.812345
val_psnr: 19.868500
val_ssim: 0.742100
peak_memory_gb: 21.4
```

You can extract the key metric from the CSV log file:
```bash
METRICS_CSV=$(find logs/decoder_stage -path "*/metrics.csv" -print | sort | tail -n 1)
PYTHONPATH=src .venv/bin/python - <<'PY'
import csv, math, os
path = os.environ["METRICS_CSV"]
rows = list(csv.DictReader(open(path, newline="")))
values = [float(row["val/loss"]) for row in rows if row.get("val/loss") not in (None, "")]
print(f"primary_metric: {values[-1]:.6f}")
PY
```

Extract all useful secondary metrics:
```bash
METRICS_CSV=$(find logs/decoder_stage -path "*/metrics.csv" -print | sort | tail -n 1)
PYTHONPATH=src .venv/bin/python - <<'PY'
import csv, os
path = os.environ["METRICS_CSV"]
rows = list(csv.DictReader(open(path, newline="")))
keys = [
    "val/loss",
    "val/l1_loss",
    "val/wavelet_loss",
    "val/mse",
    "val/mae",
    "val/rmse",
    "val/nrmse",
    "val/psnr",
    "val/ssim",
    "dynamics/grad_norm",
    "dynamics/param_norm",
    "dynamics/grad_to_param_norm",
]
for key in keys:
    values = [row[key] for row in rows if key in row and row[key] not in (None, "")]
    if values:
        print(f"{key.replace('/', '_')}: {float(values[-1]):.6f}")
PY
```

If running on CUDA, capture peak memory with a wrapper around the training command:
```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch
print(f"peak_memory_gb: {torch.cuda.max_memory_allocated() / 1024**3:.1f}")
PY
```
This standalone memory command only reports memory for its own process. For authoritative memory, wrap the training process with `/usr/bin/time -l` on macOS and convert `maximum resident set size` bytes to GB, or use `nvidia-smi --query-gpu=memory.used --format=csv` sampling during the run.

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated).

Columns:
```
commit	val/loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val/loss` achieved — use `999.000000` for crashes
3. peak memory in GB, round to .1f — use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of the experiment

Example:
```
commit	val/loss	memory_gb	status	description
a1b2c3d	0.156234	21.4	keep	baseline
b2c3d4e	0.154812	22.0	keep	add lightweight decoder residual block
```

## Documenting findings

After **every** experiment (keep, discard, or crash), you MUST write a memory note and append to results.tsv. This ensures no knowledge is lost across the loop.

### Memory notes

Write each experiment's key takeaway to memory (`~/.claude/projects/<project>/memory/`) as a markdown file. Use the `project` memory type. File naming: `exp_<tag>_<short_description>.md`.

Frontmatter format:
```yaml
---
name: exp_<tag>_<short_description>
description: <one-line summary of the experiment and its outcome>
type: project
---
```

Body must include:
- **Idea**: what was tried and why
- **Result**: `val/loss` value, comparison to baseline
- **Verdict**: keep/discard/crash — and why
- **Insight**: what was learned, such as whether decoder capacity, latent conditioning, PCGrad, or wavelet behavior was the limiting factor

Keep each note concise (5-10 lines of body). The goal is to build a cumulative research log that prevents repeating failed ideas and surfaces patterns.

Example:
```markdown
---
name: exp_may03_decoder_residual_refine
description: Lightweight residual refinement improved Stage 2 val/loss by 0.0014
type: project
---

**Idea**: Add one lightweight residual refinement block to test decoder capacity.
**Result**: val/loss 0.1548 vs baseline 0.1562, memory 21.4→22.0 GB.
**Verdict**: keep — improvement exceeds 0.001 and secondary metrics stayed stable.
**Insight**: Decoder refinement capacity is likely a bottleneck after frozen latent prediction.
```

### results.tsv

Every experiment must have a row in `results.tsv`. Do not skip crashed or discarded experiments — they are valuable negative results.

## The experiment loop

The experiment runs on a dedicated branch, e.g. `autoresearch/may03`.

LOOP FOREVER:
1. Look at the git state: the current branch/commit.
2. Modify files in **train.py scope** with an experimental idea.
3. Run the narrow relevant tests first, then the full test suite when the change is non-trivial:
   ```bash
   PYTHONPATH=src .venv/bin/python -m pytest tests -v
   ```
4. git commit.
5. Run the experiment with full log capture:
   ```bash
   RUN_TAG=<tag>-<experiment> \
   BRATS_DATA_DIR=/path/to/BraTS2023 \
   ENCODER_CHECKPOINT=/path/to/stage1.ckpt \
   /usr/bin/time -l sh -c 'PYTHONPATH=src .venv/bin/python -m jedi.train_decoder data.data_dir="$BRATS_DATA_DIR" decoder_model.encoder_checkpoint="$ENCODER_CHECKPOINT" wandb.enabled=false trainer.default_root_dir="autoresearch_runs/${RUN_TAG}" hydra.run.dir="autoresearch_runs/${RUN_TAG}/hydra"' \
     > "run_${RUN_TAG}.log" 2>&1
   ```
   Redirect everything — do NOT use `tee` or let output flood your context.
6. Read out the result:
   ```bash
   METRICS_CSV=$(find logs/decoder_stage -path "*/metrics.csv" -print | sort | tail -n 1)
   PYTHONPATH=src .venv/bin/python - <<'PY'
import csv, os
path = os.environ["METRICS_CSV"]
rows = list(csv.DictReader(open(path, newline="")))
values = [float(row["val/loss"]) for row in rows if row.get("val/loss") not in (None, "")]
print(f"primary_metric: {values[-1]:.6f}")
PY
   ```
7. If results can't be found, the run crashed. Check logs and attempt a fix. If you can't fix after a few attempts, give up on that idea.
8. Record the results in `results.tsv` (do NOT commit `results.tsv` — leave it untracked).
9. **Write a memory note** — save key findings to `memory/exp_<tag>_<description>.md` as documented in "Documenting findings" above. Every experiment gets a note.
10. If `val/loss` improved (lower is better), advance the branch.
11. If `val/loss` is equal or worse, git reset back to where you started.

**Timeout**: Each full experiment should take one configured Stage 2 run: 100 epochs unless explicitly overridden. Use the baseline wall-clock time as the expected duration for this machine/data split. If a later run exceeds 2× the baseline duration, kill it and treat it as failure unless the log shows clear forward progress and the human has approved a longer budget.

**Crashes**: If a run crashes (OOM, bug), fix simple issues and re-run. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try radical changes. The loop runs until the human interrupts you.

## Research Ideas

1. **Decoder residual refinement**: replace the current single Conv3d refinement stack in `VolumeDecoder3D` with a small residual block to test whether reconstruction quality is decoder-capacity limited.
2. **Patch-token decoder normalization**: test LayerNorm or RMS-style normalization before `to_voxels` so predicted target embeddings enter the decoder with a steadier scale.
3. **Latent-to-volume upsampling alternative**: compare the current direct patch projection against a lightweight ConvTranspose3d refinement path after patch reassembly.
4. **Predictor output stabilization for Stage 2**: add a train-time normalization or small adapter on decoder input while keeping the frozen encoder-side model unchanged.
5. **PCGrad ablation inside Stage 2**: compare current PCGrad against a direct backward on `(l1_loss + wavelet_loss) / 2` to verify whether PCGrad helps the fixed L1 + wavelet objective.
6. **Wavelet loss implementation tuning**: adjust internal wavelet decomposition behavior only within `WaveletLoss` code, while preserving the fixed external objective and no new loss-weight hyperparameters.
7. **VisualizationDecoder as diagnostic decoder**: run controlled Stage 2 experiments using `vis_decoder.yaml` and `use_cls_embedding=true` to test whether CLS embeddings retain usable spatial information.
8. **Modality conditioning stress test**: modify how `ModalityEmbedder` conditions `LatentPredictor` to see whether target-modality conditioning improves decoder validation loss without hurting latent metrics.
9. **Transformer dropout/MLP capacity sweep in code**: test small architecture changes in `LatentPredictor` and decoder-side transformer blocks, but only when checkpoint compatibility is preserved or a matching Stage 1 checkpoint is produced.
10. **Scheduler simplification**: test whether warmup/cosine scheduling beats a simpler constant LR for Stage 2 decoder-only training under the same budget.

## File Scope Reference

### prepare.py scope (DO NOT MODIFY)
| Directory/File | Role |
|---|---|
| `README.md` | Project overview, setup, data layout, training and inference commands. |
| `CLAUDE.md` | Project-specific agent instructions and invariants. |
| `pyproject.toml` | Package metadata, dependency list, console script entry points. |
| `docs/agents/domain.md` | Agent guidance for consuming domain docs. |
| `docs/agents/issue-tracker.md` | Issue tracker guidance. |
| `docs/agents/triage-labels.md` | Label vocabulary guidance. |
| `docs/superpowers/specs/2026-04-21-jedi-brats-contrast-design.md` | Historical design spec. |
| `docs/superpowers/plans/2026-04-21-jedi-brats-contrast-implementation.md` | Historical implementation plan. |
| `src/jedi/__init__.py` | Package version export. |
| `src/jedi/data/__init__.py` | Data package exports. |
| `src/jedi/data/brats.py` | BraTS dataset, modality sampling, fixed eval mapping, persistent cache, dataloader construction. |
| `src/jedi/data/transforms.py` | MONAI loading, orientation, padding/cropping, normalization, paired augmentation transforms. |
| `src/jedi/config/train_encoder.yaml` | Stage 1 Hydra config interface. |
| `src/jedi/config/train_decoder.yaml` | Stage 2 Hydra config interface. |
| `src/jedi/config/data/brats2023.yaml` | Data root, spatial size, batch size, workers, modality mapping, cache configuration. |
| `src/jedi/config/model/encoder.yaml` | Encoder-side model config interface. |
| `src/jedi/config/model/decoder.yaml` | Volume decoder config interface and Stage 1 checkpoint path. |
| `src/jedi/config/model/vis_decoder.yaml` | Visualization decoder diagnostic config interface. |
| `src/jedi/config/trainer/default.yaml` | Lightning Trainer config interface and default training budget. |
| `src/jedi/training/callbacks.py` | Fixed metric/logging callbacks for losses, dynamics, latent metrics, and reconstruction metrics. |
| `src/jedi/training/logging.py` | Resolved Hydra config logging and WandB config update infrastructure. |
| `src/jedi/training/trainer_config.py` | Dataclass adapter for supported Lightning Trainer fields. |
| `src/jedi/utils.py` | Encoder-side checkpoint loading utility. |
| `src/jedi/infer.py` | Inference CLI and checkpoint loading path. |
| `tests/test_brats_dataset.py` | Dataset, cache, and dataloader behavior tests. |
| `tests/test_decoder_step.py` | Stage 2 training, freezing, PCGrad, scheduler/Trainer smoke tests. |
| `tests/test_encoder_step.py` | Stage 1 forward/loss smoke tests. |
| `tests/test_flash_attn.py` | FlashAttention fallback and attention shape tests. |
| `tests/test_metric_callbacks.py` | Metric callback coverage for logged metric names. |
| `tests/test_trainer_config.py` | Trainer config and total-step estimation tests. |
| `tests/test_training_logging.py` | Resolved config logging tests. |
| `tests/test_transforms.py` | Transform and augmentation behavior tests. |
| `tests/test_vis_decoder.py` | Visualization decoder and CLS diagnostic behavior tests. |
| `tests/test_vit3d_shapes.py` | 3D ViT shape test. |
| `tests/test_wavelet_loss.py` | Wavelet loss correctness and gradient tests. |

### train.py scope (MODIFIABLE)
| Directory/File | Role |
|---|---|
| `src/jedi/models/__init__.py` | Model exports and model composition surface. |
| `src/jedi/models/components.py` | MLP projector/pred-proj blocks and modality embedder. |
| `src/jedi/models/decoder3d.py` | Patch-token-to-volume decoder architecture. |
| `src/jedi/models/jepa.py` | CrossModalityJEPA wrapper and source-to-target latent prediction path. |
| `src/jedi/models/predictor.py` | Conditional latent predictor and AdaLN-style transformer blocks. |
| `src/jedi/models/regularizers.py` | SIGReg latent regularizer. |
| `src/jedi/models/transformer.py` | Transformer attention/feed-forward primitives and FlashAttention fallback path. |
| `src/jedi/models/vis_decoder.py` | Diagnostic CLS-to-volume decoder architecture. |
| `src/jedi/models/vit3d.py` | 3D ViT encoder architecture. |
| `src/jedi/models/wavelet_loss.py` | Wavelet-domain reconstruction loss implementation. |
| `src/jedi/training/__init__.py` | Training module exports. |
| `src/jedi/training/encoder_module.py` | Stage 1 LightningModule, latent prediction objective, optimizer hook. |
| `src/jedi/training/decoder_module.py` | Stage 2 LightningModule, frozen encoder-side path, decoder objective, PCGrad manual optimization. |
| `src/jedi/training/optim.py` | AdamW parameter grouping and LR scheduler construction. |
| `src/jedi/training/schedule.py` | Total step estimation used by scheduler setup. |
| `src/jedi/train_encoder.py` | Stage 1 training entry point and runtime model/module composition. |
| `src/jedi/train_decoder.py` | Stage 2 training entry point and runtime model/module composition. |
