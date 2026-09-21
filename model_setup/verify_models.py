#!/usr/bin/env python3
"""Check downloaded model trees and emit a reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED = {
    "Qwen3-Omni-30B-A3B-Instruct": ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"),
    "MiniCPM-o-4_5": ("openbmb/MiniCPM-o-4_5", "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"),
    "gemma-4-12B-it": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "audio-flamingo-3-hf": ("nvidia/audio-flamingo-3-hf", "7d4bae64ee29878af6504ae6f6bb3e40492838ad"),
    "Qwen3-Omni-30B-A3B-Thinking": ("Qwen/Qwen3-Omni-30B-A3B-Thinking", "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"),
    "Step-Audio-2-mini": ("stepfun-ai/Step-Audio-2-mini", "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36"),
    "Kimi-Audio-7B-Instruct": ("moonshotai/Kimi-Audio-7B-Instruct", "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"),
    "Phi-4-multimodal-instruct": ("microsoft/Phi-4-multimodal-instruct", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"),
    "MiMo-Audio-7B-Instruct": ("XiaomiMiMo/MiMo-Audio-7B-Instruct", "c359441c22c2a1c74be5f99a91e83392680e9cc8"),
    "MiMo-Audio-Tokenizer": ("XiaomiMiMo/MiMo-Audio-Tokenizer", "5df9914f72d3acda1320d7fecde7d91622edb0c1"),
}
CORE_NAMES = {
    "Qwen3-Omni-30B-A3B-Instruct",
    "MiniCPM-o-4_5",
    "gemma-4-12B-it",
    "audio-flamingo-3-hf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_model(path: Path, repo_id: str, revision: str) -> dict:
    files = [item for item in path.rglob("*") if item.is_file()]
    zero_weights = [str(item.relative_to(path)) for item in files if item.suffix == ".safetensors" and item.stat().st_size == 0]
    broken_links = [str(item.relative_to(path)) for item in path.rglob("*") if item.is_symlink() and not item.exists()]
    indices = list(path.rglob("model.safetensors.index.json"))
    missing_shards: list[str] = []
    for index in indices:
        payload = json.loads(index.read_text(encoding="utf-8"))
        for shard in sorted(set(payload.get("weight_map", {}).values())):
            if not (index.parent / shard).is_file():
                missing_shards.append(str((index.parent / shard).relative_to(path)))
    configs = []
    for name in ("config.json", "processor_config.json", "preprocessor_config.json", "generation_config.json"):
        for config in path.rglob(name):
            configs.append({"path": str(config.relative_to(path)), "sha256": sha256(config)})
    return {
        "repo_id": repo_id,
        "revision": revision,
        "path": str(path),
        "present": path.is_dir(),
        "bytes": sum(item.stat().st_size for item in files),
        "file_count": len(files),
        "weight_files": sorted(str(item.relative_to(path)) for item in files if item.suffix in {".safetensors", ".bin"}),
        "index_files": sorted(str(item.relative_to(path)) for item in indices),
        "config_hashes": sorted(configs, key=lambda row: row["path"]),
        "zero_byte_weights": zero_weights,
        "broken_symlinks": broken_links,
        "missing_index_shards": missing_shards,
        "valid": path.is_dir() and bool(files) and not zero_weights and not broken_links and not missing_shards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("MODEL_ROOT", str(Path.home() / "audio_model_setup" / "models"))))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group", choices=("core", "candidates", "all"), default="all")
    args = parser.parse_args()
    selected = {
        name: metadata
        for name, metadata in EXPECTED.items()
        if args.group == "all"
        or (args.group == "core" and name in CORE_NAMES)
        or (args.group == "candidates" and name not in CORE_NAMES)
    }
    report = {
        "schema": "audio_model_setup_manifest_v1",
        "group": args.group,
        "models": [inspect_model(args.model_root / name, *metadata) for name, metadata in selected.items()],
    }
    report["all_valid"] = all(row["valid"] for row in report["models"])
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
