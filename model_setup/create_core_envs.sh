#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MAMBA="${AUDIO_MODEL_SETUP_ROOT}/tools/micromamba-bin/micromamba"
if [[ ! -x "$MAMBA" ]]; then
  echo "Missing micromamba at ${MAMBA}" >&2
  exit 2
fi

create_base() {
  local prefix="$1"
  if [[ ! -x "${prefix}/bin/python" ]]; then
    "$MAMBA" create -y -p "$prefix" -c conda-forge \
      python=3.11 pip ffmpeg sox libsndfile ninja
  fi
}

install_modern() {
  local prefix="${VENV_ROOT}/modern-omni"
  create_base "$prefix"
  "${prefix}/bin/python" -m pip install --upgrade pip
  "${prefix}/bin/python" -m pip install \
    torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
  "${prefix}/bin/python" -m pip install \
    transformers==5.12.1 accelerate librosa soundfile scipy pillow \
    qwen-omni-utils
  "${prefix}/bin/python" -m pip check
  "${prefix}/bin/python" -m pip freeze > "${AUDIO_MODEL_SETUP_ROOT}/manifests/modern-omni-pip-freeze.txt"
}

install_minicpm() {
  local prefix="${VENV_ROOT}/minicpm451"
  create_base "$prefix"
  "${prefix}/bin/python" -m pip install --upgrade pip
  "${prefix}/bin/python" -m pip install \
    torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
  "${prefix}/bin/python" -m pip install \
    transformers==4.51.0 accelerate 'minicpmo-utils[all]>=1.0.5'
  # librosa 0.9 imports pkg_resources, which setuptools 81+ removed.
  "${prefix}/bin/python" -m pip install setuptools==80.9.0
  check_output="$("${prefix}/bin/python" -m pip check 2>&1)" || {
    if [[ "$check_output" != "decord 0.6.0 is not supported on this platform" ]]; then
      echo "$check_output" >&2
      return 1
    fi
    echo "Known metadata warning (runtime import verified separately): $check_output"
  }
  "${prefix}/bin/python" -m pip freeze > "${AUDIO_MODEL_SETUP_ROOT}/manifests/minicpm451-pip-freeze.txt"
}

case "${1:-all}" in
  modern) install_modern ;;
  minicpm) install_minicpm ;;
  all) install_modern; install_minicpm ;;
  *) echo "Usage: $0 {all|modern|minicpm}" >&2; exit 2 ;;
esac
