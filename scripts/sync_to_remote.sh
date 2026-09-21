#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <user@host> <remote-directory>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="$1:$2/"

python3 "$ROOT/scripts/validate_dataset.py"
rsync -ah --info=progress2 --partial \
  --exclude '.git/' --exclude '__pycache__/' \
  "$ROOT/" "$REMOTE"
echo "Transferred to $REMOTE"
