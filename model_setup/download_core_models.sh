#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

HF_BIN="${VENV_ROOT}/hf-download/bin/hf"
if [[ ! -x "$HF_BIN" ]]; then
  echo "Missing ${HF_BIN}; create the hf-download environment first." >&2
  exit 2
fi

download() {
  local repo="$1"
  local revision="$2"
  local target="$3"
  "$HF_BIN" download "$repo" --revision "$revision" --local-dir "${MODEL_ROOT}/${target}"
}

case "${1:-core}" in
  qwen3)
    download Qwen/Qwen3-Omni-30B-A3B-Instruct 26291f793822fb6be9555850f06dfe95f2d7e695 Qwen3-Omni-30B-A3B-Instruct
    ;;
  minicpm)
    download openbmb/MiniCPM-o-4_5 073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1 MiniCPM-o-4_5
    ;;
  gemma4)
    download google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 gemma-4-12B-it
    ;;
  af3)
    download nvidia/audio-flamingo-3-hf 7d4bae64ee29878af6504ae6f6bb3e40492838ad audio-flamingo-3-hf
    ;;
  core)
    download Qwen/Qwen3-Omni-30B-A3B-Instruct 26291f793822fb6be9555850f06dfe95f2d7e695 Qwen3-Omni-30B-A3B-Instruct
    download openbmb/MiniCPM-o-4_5 073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1 MiniCPM-o-4_5
    download google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 gemma-4-12B-it
    download nvidia/audio-flamingo-3-hf 7d4bae64ee29878af6504ae6f6bb3e40492838ad audio-flamingo-3-hf
    ;;
  *)
    echo "Usage: $0 {core|qwen3|minicpm|gemma4|af3}" >&2
    exit 2
    ;;
esac
