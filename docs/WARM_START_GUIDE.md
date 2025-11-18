# ♻️ Warm-Start & Checkpoint Guide

## Why this matters
Phase 2 now relies on building from previous breakthroughs (e.g. Track G2 → G2+ → Track H). Every runner can persist lightweight JSON checkpoints that capture policy weights plus metadata, making it trivial to:

- Resume a partially completed phase after interruptions.
- Feed the best Phase G agent into G2+, G3.1, or Track H memory experiments.
- Share reproducible agents between collaborators without exporting giant tensors.

Use this guide whenever you kick off a new phase or resume an existing one.

---

## 1. Where checkpoints live

| Phase / Track | Default Save Path |
|---------------|------------------|
| Track G Phase G2 | `logs/track_g/checkpoints/phase_g2_latest.json` |
| Track G Phase G3 | `logs/track_g/checkpoints/phase_g3_latest.json` |
| Track G Phase G2+ | `logs/track_g/checkpoints/phase_g2plus_latest.json` |
| Track H Memory (per architecture) | `logs/track_h/checkpoints/<arch>_agent_latest.json` |

Each file is plain JSON: it stores the agent class, timestamp, metadata, and weight matrices, so you can version-control it or inspect diffs if needed.

---

## 2. Enabling warm-starts in configs

Every phase config now exposes a `warm_start` block:

```yaml
phase_g2:
  ...
  warm_start:
    load_path: null  # set to an existing checkpoint to resume
    save_path: "logs/track_g/checkpoints/phase_g2_latest.json"
    metadata:
      description: "Phase G2 agent snapshot for reuse"
```

- **To resume**: set `load_path` to the latest checkpoint file.
- **To produce a checkpoint**: keep `save_path` pointing at your preferred file; the runner overwrites it whenever a phase completes.

Track H inherits from Track G by default:

```yaml
track_h:
  warm_start:
    load_path: "logs/track_g/checkpoints/phase_g2_latest.json"
```

Each architecture also sets a unique `save_path`, so you keep LSTM/GRU/attention agents separate.

---

## 3. Typical workflow

1. **Phase G2**  
   Run `make track-g PHASE=g2 CONFIG=fre/configs/track_g_phase_g2.yaml`.  
   After completion, `logs/track_g/checkpoints/phase_g2_latest.json` is created.

2. **Phase G2+ continuation**  
   Ensure `phase_g2plus.warm_start.load_path` points to the G2 checkpoint, then run `make track-g PHASE=g2plus CONFIG=fre/configs/track_g2plus.yaml`. The run resumes from Episode 54 weights and saves a new `phase_g2plus_latest.json` for downstream phases.

3. **Track H memory integration**  
   `track_h.warm_start.load_path` already references the G2 checkpoint. Launch with `make track-h CONFIG=fre/configs/track_h_memory.yaml`; each architecture saves its own file (e.g. `lstm_agent_latest.json`) so you can iterate on architectures independently.

---

## 4. Tips & troubleshooting

- **Versioning**: add these JSON files to `.gitignore` if you only need them locally, or commit them to a dedicated branch when sharing agents across sites.
- **Multiple checkpoints**: duplicate the `save_path` before launching exploratory runs (e.g. `phase_g2_latest_best.json`) so you always keep the best-known agent.
- **Sanity check**: the new `tests/test_track_g_runner.py` suite verifies JSON safety and checkpoint round-trips—run `pytest tests/test_track_g_runner.py` after modifying runner internals.
- **CLI helpers**: use `make checkpoint-list DIR=…`, `make checkpoint-info CHECKPOINT=…`, or `make checkpoint-copy SRC=… DEST=… OVERWRITE=1` (powered by `scripts/checkpoint_tool.py`) to inspect and archive agents without hand-editing JSON.
- **Config snapshots**: every checkpoint embeds the exact YAML used. Extract it with `poetry run python scripts/checkpoint_tool.py extract-config --path ... --output restored.yaml` if you need to reproduce outside the repo.
- **Config registry**: label configs with `make config-register CONFIG=... LABEL="..."` so checkpoint metadata can show a human-readable name; use `make config-lookup HASH=...` to audit historical runs.
- **Config diffs**: `make config-diff A=... B=...` (powered by `scripts/config_registry.py diff`) highlights YAML changes between experiments, helping you justify why a checkpoint needed to change.
- **Registry formatting**: `nix flake check` runs `scripts/validate_registry.py` to ensure `configs/config_registry.json` stays canonical (sorted keys, 2-space indent). If it fails, reformat the file and re-run.
- **On-the-fly overrides**: both `fre/track_g_runner.py` and `fre/track_h_runner.py` accept `--warm-start-load` / `--warm-start-save`, and the `make track-g` / `make track-h` targets expose these via `WARM_LOAD` / `WARM_SAVE` env vars. This lets you test different checkpoints without touching YAML.
- **Dry runs**: add `--dry-run` (or `DRY_RUN=1` when using Make) to verify configs + checkpoint paths before launching 10-hour experiments.
- **Real-time monitoring**: enable `experiment.log_jsonl.enabled` to stream per-episode metrics, then use `make log-tail PATH=...` or `make log-validate PATH=...` (powered by `scripts/log_tool.py`) to watch or sanity-check the logs.
- **Archival**: after a successful run, execute `make archive-artifacts CHECKPOINT=... LOG=... CONFIG=...` to produce a compressed bundle with checkpoint, log, config snapshot, and hashes for reproducibility. If you've registered the config hash, the archive metadata also records your human-readable label so future verifiers know exactly which config was used.
- **Checkpoint snapshots in archives**: when creating archives, the tool now stores *both* the config file you pass and the embedded checkpoint snapshot. This means bundles always contain a human-readable YAML even if the original config file disappears.
- **Archive verification**: share `.tar.gz` bundles with confidence—everyone can run `make archive-verify ARCHIVE=...` to confirm hashes recorded in `metadata.json` match the embedded files.
- **Archive versus live config diff**: run `poetry run python scripts/archive_tool.py diff --archive bundle.tar.gz --config fre/configs/...` to confirm the archived snapshot still matches your current config (ideal before publishing a bundle).
- **Archive summaries**: `poetry run python scripts/archive_tool.py summary --archive bundle.tar.gz --markdown --markdown-path release.md` produces a Markdown table for release notes; rerun with `--check-markdown` to ensure existing notes still match the archive.
- **Repro provenance**: every checkpoint automatically stores the config path, config SHA-256, and git commit. The runner validates that loaded checkpoints match the currently provided config—if hashes differ you'll get a clear error prompting you to regenerate or consciously override.
- **Intentional overrides**: if you *must* reuse a checkpoint with a different config (e.g. for ablation comparisons), set `warm_start.allow_mismatch: true` in the config or pass `ALLOW_MISMATCH=1 make track-g ...`. This prints a loud warning and skips the hash guard—use sparingly and document the rationale.

With these warm-start hooks, you should never have to rebuild a consciousness-grade agent from scratch again. Happy threshold crossing! 🌊
