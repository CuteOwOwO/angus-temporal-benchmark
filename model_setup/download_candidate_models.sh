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

case "${1:-candidates}" in
  qwen3-thinking)
    download Qwen/Qwen3-Omni-30B-A3B-Thinking 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b Qwen3-Omni-30B-A3B-Thinking ;;
  step)
    download stepfun-ai/Step-Audio-2-mini e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36 Step-Audio-2-mini ;;
  kimi)
    download moonshotai/Kimi-Audio-7B-Instruct 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b Kimi-Audio-7B-Instruct ;;
  phi4)
    download microsoft/Phi-4-multimodal-instruct 93f923e1a7727d1c4f446756212d9d3e8fcc5d81 Phi-4-multimodal-instruct ;;
  mimo)
    download XiaomiMiMo/MiMo-Audio-7B-Instruct c359441c22c2a1c74be5f99a91e83392680e9cc8 MiMo-Audio-7B-Instruct
    download XiaomiMiMo/MiMo-Audio-Tokenizer 5df9914f72d3acda1320d7fecde7d91622edb0c1 MiMo-Audio-Tokenizer ;;
  candidates)
    download Qwen/Qwen3-Omni-30B-A3B-Thinking 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b Qwen3-Omni-30B-A3B-Thinking
    download stepfun-ai/Step-Audio-2-mini e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36 Step-Audio-2-mini
    download moonshotai/Kimi-Audio-7B-Instruct 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b Kimi-Audio-7B-Instruct
    download microsoft/Phi-4-multimodal-instruct 93f923e1a7727d1c4f446756212d9d3e8fcc5d81 Phi-4-multimodal-instruct
    download XiaomiMiMo/MiMo-Audio-7B-Instruct c359441c22c2a1c74be5f99a91e83392680e9cc8 MiMo-Audio-7B-Instruct
    download XiaomiMiMo/MiMo-Audio-Tokenizer 5df9914f72d3acda1320d7fecde7d91622edb0c1 MiMo-Audio-Tokenizer ;;
  *) echo "Usage: $0 {candidates|qwen3-thinking|step|kimi|phi4|mimo}" >&2; exit 2 ;;
esac
