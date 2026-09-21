#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MAMBA="${AUDIO_MODEL_SETUP_ROOT}/tools/micromamba-bin/micromamba"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"

create_base() {
  local prefix="$1"
  local python_version="$2"
  if [[ ! -x "${prefix}/bin/python" ]]; then
    "$MAMBA" create -y -p "$prefix" -c conda-forge \
      "python=${python_version}" pip ffmpeg sox libsndfile ninja
  fi
}

install_torch26() {
  local prefix="$1"
  "${prefix}/bin/python" -m pip install \
    sympy==1.13.1 'mpmath<1.4' typing_extensions filelock fsspec \
    jinja2 networkx numpy pillow
  "${prefix}/bin/python" -m pip install \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url "$PYTORCH_INDEX"
}

freeze() {
  local name="$1"
  "${VENV_ROOT}/${name}/bin/python" -m pip freeze > "${AUDIO_MODEL_SETUP_ROOT}/manifests/${name}-pip-freeze.txt"
}

install_step() {
  local prefix="${VENV_ROOT}/step449"
  create_base "$prefix" 3.10
  install_torch26 "$prefix"
  "${prefix}/bin/python" -m pip install \
    transformers==4.49.0 librosa onnxruntime s3tokenizer diffusers hyperpyyaml
  "${prefix}/bin/python" -m pip check
  freeze step449
}

install_phi() {
  local prefix="${VENV_ROOT}/phi448"
  create_base "$prefix" 3.10
  install_torch26 "$prefix"
  "${prefix}/bin/python" -m pip install \
    transformers==4.48.2 accelerate==1.3.0 soundfile==0.13.1 \
    pillow==11.1.0 scipy==1.15.2 backoff==2.2.1 peft==0.13.2
  "${prefix}/bin/python" -m pip check
  freeze phi448
}

install_kimi() {
  local prefix="${VENV_ROOT}/kimi"
  create_base "$prefix" 3.11
  install_torch26 "$prefix"
  # flash-attn is omitted because this host has no CUDA toolkit/nvcc.
  "${prefix}/bin/python" -m pip install \
    -r <(grep -Ev '^(torch|torchaudio|flash_attn)==' "${RUNTIME_ROOT}/Kimi-Audio/requirements.txt")
  "${prefix}/bin/python" -m pip install --no-deps -e "${RUNTIME_ROOT}/Kimi-Audio"
  freeze kimi
}

install_mimo() {
  local prefix="${VENV_ROOT}/mimo"
  create_base "$prefix" 3.11
  install_torch26 "$prefix"
  "${prefix}/bin/python" -m pip install \
    -r <(grep -Ev '^(torch|torchaudio|triton)==' "${RUNTIME_ROOT}/MiMo-Audio/requirements.txt")
  "${prefix}/bin/python" -m pip check
  freeze mimo
}

install_gemma3n() {
  local prefix="${VENV_ROOT}/gemma3n453"
  create_base "$prefix" 3.11
  "${prefix}/bin/python" -m pip install \
    torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url "$PYTORCH_INDEX"
  "${prefix}/bin/python" -m pip install \
    transformers==4.53.0 accelerate librosa soundfile pillow
  "${prefix}/bin/python" -m pip check
  freeze gemma3n453
}

case "${1:-all}" in
  step) install_step ;;
  phi) install_phi ;;
  kimi) install_kimi ;;
  mimo) install_mimo ;;
  gemma3n) install_gemma3n ;;
  all) install_step; install_phi; install_kimi; install_mimo; install_gemma3n ;;
  *) echo "Usage: $0 {all|step|phi|kimi|mimo|gemma3n}" >&2; exit 2 ;;
esac
