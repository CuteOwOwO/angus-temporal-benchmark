# Angus Temporal Audio Benchmark

Portable release for running the temporal audio QA benchmark on a separate GPU
machine.

## Contents

- `benchmark/audio/`: 112 mono 44.1 kHz WAV files.
- `benchmark/data/qa.public.json`: 253 evaluation questions.
- `benchmark/data/qa.private.json`: hidden answer key for scoring.
- `benchmark/data/benchmark_manifest.public.json`: public audio manifest.
- `benchmark/data/composition_manifest.private.json`: construction provenance.
- `model_setup/`: model download and environment helpers.
- `generation/`: frozen generation and assembly code for provenance.
- `web_review/`: local composition and QA review interface.
- `docs/REMOTE_MACHINE_MODEL_SETUP.md`: GPU machine setup notes.
- `scripts/validate_dataset.py`: standalone integrity validation.

The benchmark has exactly four options per question and one to three questions
per audio file. Keep `qa.private.json` away from model prompts and generated
outputs.

## Clone On The GPU Machine

```bash
git lfs install
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd temporal-benchmark
git lfs pull
python3 scripts/validate_dataset.py
```

Expected validation result:

```text
PASS: 112 audio files, 253 QA items
```

## Review Website

Start the bundled review interface with no extra dependencies:

```bash
python3 web_review/server.py
```

Then open <http://127.0.0.1:8775>. It provides separate views for all 112
compositions and all 253 QA items. Verdicts and notes are written to
`web_review/data/`; commit those small JSON files to move review progress
between machines.

Models and Hugging Face caches should live on the GPU machine's large data
disk, not inside this repository. Follow `docs/REMOTE_MACHINE_MODEL_SETUP.md`.
Override the default storage location before running setup scripts:

```bash
export AUDIO_MODEL_SETUP_ROOT=/data/$USER/audio_model_setup
source model_setup/env.sh
```

The files under `generation/` document exactly how this release was produced.
The final benchmark itself is self-contained, but rebuilding the parent audio
sets from scratch additionally requires the reviewed ESC-50 source registries
from the original research workspace.

## First Push

Create an empty private GitHub repository, then run:

```bash
./scripts/init_git_repo.sh git@github.com:<OWNER>/<REPOSITORY>.git
git push -u origin main
```

WAV files are tracked with Git LFS. JSON, Python, shell, and documentation files
remain normal Git objects.

For a one-off transfer without GitHub:

```bash
./scripts/sync_to_remote.sh <USER@HOST> /data/temporal-benchmark
```

## Data License

The audio is derived from ESC-50 and is for non-commercial research use under
CC BY-NC 3.0. See `licenses/ESC-50-LICENSE` for the dataset license and original
clip attribution. Cite Piczak (2015) when publishing results.
