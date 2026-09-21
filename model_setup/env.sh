#!/usr/bin/env bash

# Shared locations for the remote-machine model setup. Keep large artifacts
# outside the project tree so benchmark packaging cannot include them.
export AUDIO_MODEL_SETUP_ROOT="${AUDIO_MODEL_SETUP_ROOT:-${HOME}/audio_model_setup}"
export MODEL_ROOT="${MODEL_ROOT:-${AUDIO_MODEL_SETUP_ROOT}/models}"
export VENV_ROOT="${VENV_ROOT:-${AUDIO_MODEL_SETUP_ROOT}/venvs}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${AUDIO_MODEL_SETUP_ROOT}/runtime}"
export HF_HOME="${HF_HOME:-${AUDIO_MODEL_SETUP_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TMPDIR="${TMPDIR:-${AUDIO_MODEL_SETUP_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${AUDIO_MODEL_SETUP_ROOT}/cache}"
export PYTHONNOUSERSITE=1

mkdir -p "$MODEL_ROOT" "$VENV_ROOT" "$RUNTIME_ROOT" \
  "$HF_HUB_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" \
  "${AUDIO_MODEL_SETUP_ROOT}/logs" "${AUDIO_MODEL_SETUP_ROOT}/manifests"
