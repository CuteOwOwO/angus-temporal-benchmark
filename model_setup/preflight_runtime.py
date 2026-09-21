#!/usr/bin/env python3
"""Offline CUDA, audio decoding, config, and processor preflight checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL_DIRS = {
    "qwen3": "Qwen3-Omni-30B-A3B-Instruct",
    "qwen3-thinking": "Qwen3-Omni-30B-A3B-Thinking",
    "minicpm": "MiniCPM-o-4_5",
    "gemma4": "gemma-4-12B-it",
    "af3": "audio-flamingo-3-hf",
    "step": "Step-Audio-2-mini",
    "kimi": "Kimi-Audio-7B-Instruct",
    "phi4": "Phi-4-multimodal-instruct",
    "mimo": "MiMo-Audio-7B-Instruct",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_DIRS), required=True)
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("MODEL_ROOT", str(Path.home() / "audio_model_setup" / "models"))))
    parser.add_argument("--audio", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark" / "audio" / "esc50div13_012cd2b3bdc57f.wav")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import librosa
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    waveform, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    report = {
        "schema": "audio_model_runtime_preflight_v1",
        "offline": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [],
        "audio": {
            "path": str(args.audio),
            "sample_rate": sample_rate,
            "samples": int(waveform.shape[0]),
            "duration_sec": round(float(waveform.shape[0] / sample_rate), 3),
        },
        "models": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["cuda_devices"].append({
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory": properties.total_memory,
            })
        probe = torch.ones(1, device="cuda:0")
        report["cuda_tensor_ok"] = bool(probe.item() == 1)
        del probe
        torch.cuda.empty_cache()
    else:
        report["cuda_tensor_ok"] = False

    for name in args.models:
        path = args.model_root / MODEL_DIRS[name]
        row = {"name": name, "path": str(path), "config_ok": False, "processor_ok": False}
        try:
            config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            row.update(config_ok=True, model_type=getattr(config, "model_type", None))
        except Exception as exc:  # Keep every model result in the final report.
            row["config_error"] = f"{type(exc).__name__}: {exc}"
        try:
            processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            row.update(processor_ok=True, processor_class=type(processor).__name__)
        except Exception as exc:
            row["processor_error"] = f"{type(exc).__name__}: {exc}"
        report["models"].append(row)

    report["valid"] = (
        report["cuda_tensor_ok"]
        and sample_rate == 16000
        and all(row["config_ok"] and row["processor_ok"] for row in report["models"])
    )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
