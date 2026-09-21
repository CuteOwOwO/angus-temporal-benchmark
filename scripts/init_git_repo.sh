#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <git-remote-url>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v git-lfs >/dev/null || { echo "git-lfs is required" >&2; exit 1; }
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi
git lfs install --local
git lfs track 'benchmark/audio/*.wav'
python3 scripts/validate_dataset.py
git add .

if ! git config user.name >/dev/null || ! git config user.email >/dev/null; then
  echo "Set git user.name and user.email, then run: git commit -m 'Initial temporal benchmark release'" >&2
  exit 1
fi

git commit -m "Initial temporal benchmark release"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$1"
else
  git remote add origin "$1"
fi
echo "Ready. Push with: git push -u origin main"
