#!/usr/bin/env python3
"""Build the deterministic repeated-event v2 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TC = ROOT.parent
PROJECT = TC.parent
PARENT = TC / "esc50_temporal_scaled_v1"
PARENT_RELEASE = PARENT / "release"
sys.path.insert(0, str(PARENT))
import build_scaled as base  # noqa: E402


OUT = ROOT / "release"
SEED = 20260921_502
SOURCE_CAP = 3
PATTERNS = ("ABACD", "ABCAD", "ABCDA", "ABCBD", "ABCDB", "ABCDC")
EXTRA_BANNED = {frozenset(("engine", "vacuum_cleaner"))}
BANNED_PAIRS = base.BANNED_PAIRS | EXTRA_BANNED


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(*parts: object) -> int:
    payload = json.dumps([SEED, *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def pattern_repeat(pattern: str) -> tuple[str, tuple[int, int]]:
    counts = Counter(pattern)
    repeated = next(symbol for symbol, count in counts.items() if count == 2)
    positions = tuple(index for index, symbol in enumerate(pattern) if symbol == repeated)
    return repeated, positions  # type: ignore[return-value]


def categories_valid(categories: list[str], mode: str) -> bool:
    value = set(categories)
    if len(value) != 4 or any(pair <= value for pair in BANNED_PAIRS):
        return False
    containing_scenes = [name for name, members in base.SCENES.items() if value <= members]
    return bool(containing_scenes) if mode == "coherent_scene" else not containing_scenes


def assign_sources(
    category_sequence: list[str],
    by_category: dict[str, list[dict]],
    uses: Counter,
    plan_key: tuple,
) -> list[dict] | None:
    assigned: list[dict] = []
    local_ids: set[str] = set()
    for position, category in enumerate(category_sequence):
        choices = [
            row for row in by_category[category]
            if uses[row["candidate_id"]] < SOURCE_CAP
            and row["candidate_id"] not in local_ids
            and row["audio_sha256"] not in {item["audio_sha256"] for item in assigned}
        ]
        choices.sort(key=lambda row: (
            uses[row["candidate_id"]],
            stable("source", plan_key, position, row["candidate_id"]),
        ))
        if not choices:
            return None
        chosen = choices[0]
        assigned.append(chosen)
        local_ids.add(chosen["candidate_id"])
        uses[chosen["candidate_id"]] += 1
    return assigned


def plan_compositions(sources: list[dict]) -> list[dict]:
    eligible = [row for row in sources if row["eligible_for_generation"]]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_category[row["category"]].append(row)
    repeatable = sorted(category for category, rows in by_category.items() if len(rows) >= 2)
    if len(repeatable) != 35:
        raise ValueError(f"expected 35 repeatable categories, found {len(repeatable)}")

    pattern_slots = [pattern for pattern in PATTERNS for _ in range(7)]
    slots = [
        (pattern, "coherent_scene" if index % 2 == 0 else "cross_context_stress")
        for index, pattern in enumerate(pattern_slots)
    ]

    for attempt in range(2000):
        rng = random.Random(SEED + attempt)
        shuffled_slots = list(slots)
        rng.shuffle(shuffled_slots)
        repeated_roles = list(repeatable)
        repeated_roles.extend(rng.sample(repeatable, 7))
        rng.shuffle(repeated_roles)
        uses: Counter = Counter()
        category_uses: Counter = Counter()
        seen_sets: set[frozenset[str]] = set()
        plans: list[dict] = []
        failed = False

        for ordinal, ((pattern, mode), repeated_category) in enumerate(zip(shuffled_slots, repeated_roles), 1):
            repeated_symbol, repeat_positions = pattern_repeat(pattern)
            found: tuple[list[str], dict[str, str], list[dict], str] | None = None
            for trial in range(2500):
                if mode == "coherent_scene":
                    scene_names = [name for name, members in base.SCENES.items() if repeated_category in members]
                    if not scene_names:
                        continue
                    scene = scene_names[stable("scene", attempt, ordinal, trial) % len(scene_names)]
                    pool = sorted(base.SCENES[scene] - {repeated_category})
                else:
                    scene = "cross_context_stress"
                    pool = sorted(set(by_category) - {repeated_category})
                pool = sorted(pool, key=lambda category: (
                    category_uses[category], stable("category", attempt, ordinal, trial, category)
                ))
                window = pool[:min(len(pool), 15)]
                if len(window) < 3:
                    continue
                singleton_categories = rng.sample(window, 3)
                unique_categories = [repeated_category, *singleton_categories]
                category_set = frozenset(unique_categories)
                if category_set in seen_sets or not categories_valid(unique_categories, mode):
                    continue
                symbols = sorted(set(pattern) - {repeated_symbol})
                rng.shuffle(singleton_categories)
                symbol_map = {repeated_symbol: repeated_category, **dict(zip(symbols, singleton_categories))}
                sequence = [symbol_map[symbol] for symbol in pattern]
                trial_uses = Counter(uses)
                assigned = assign_sources(sequence, by_category, trial_uses, (attempt, ordinal, pattern))
                if assigned is None:
                    continue
                found = sequence, symbol_map, assigned, scene
                uses = trial_uses
                break
            if found is None:
                failed = True
                break
            sequence, symbol_map, assigned, scene = found
            category_uses.update(sequence)
            seen_sets.add(frozenset(sequence))
            plans.append({
                "ordinal": ordinal,
                "pattern_id": pattern,
                "repeat_category": repeated_category,
                "repeat_positions": list(repeat_positions),
                "scene_mode": mode,
                "scene_private": scene,
                "category_sequence": sequence,
                "symbol_map": symbol_map,
                "sources": assigned,
            })
        if not failed and len(plans) == 42 and max(uses.values()) <= SOURCE_CAP:
            return plans
    raise RuntimeError("unable to plan 42 repeated-event compositions within source cap")


def validate_release(root: Path) -> None:
    private = load(root / "data" / "composition_manifest.private.json")
    public = load(root / "data" / "benchmark_manifest.public.json")
    sources = load(root / "data" / "source_manifest.private.json")
    report = load(root / "data" / "build_report.json")
    if len(private) != 42 or len(public) != 42 or len(sources) != 96:
        raise ValueError("release size failure")
    if Counter(row["pattern_id_private"] for row in private) != Counter({pattern: 7 for pattern in PATTERNS}):
        raise ValueError("pattern balance failure")
    if Counter(row["scene_mode_private"] for row in private) != Counter({"coherent_scene": 21, "cross_context_stress": 21}):
        raise ValueError("scene balance failure")
    expected_parent_hash = digest(PARENT_RELEASE / "data" / "composition_manifest.private.json")
    if report["parent_manifest_sha256"] != expected_parent_hash:
        raise ValueError("parent manifest provenance failure")
    eligible = {row["candidate_id"]: row for row in sources if row["eligible_for_generation"]}
    uses: Counter = Counter()
    repeated_roles: Counter = Counter()
    seen_sets: set[frozenset[str]] = set()
    public_by_id = {row["sample_id"]: row for row in public}
    if len(public_by_id) != 42 or any(set(row) != {"sample_id", "duration_sec", "audio_file", "audio_sha256"} for row in public):
        raise ValueError("public manifest failure")

    for row in private:
        intervals = row["intervals"]
        categories = [item["category"] for item in intervals]
        counts = Counter(categories)
        if len(intervals) != 5 or sorted(counts.values()) != [1, 1, 1, 2]:
            raise ValueError("invalid repeated-event cardinality")
        repeated = next(category for category, count in counts.items() if count == 2)
        positions = [index for index, category in enumerate(categories) if category == repeated]
        if repeated != row["repeat_category_private"] or positions != row["repeat_positions_private"] or positions[1] - positions[0] < 2:
            raise ValueError("repeat metadata failure")
        expected_pattern = row["pattern_id_private"]
        if pattern_repeat(expected_pattern)[1] != tuple(positions):
            raise ValueError("pattern position failure")
        if not categories_valid(categories, row["scene_mode_private"]):
            raise ValueError("scene or banned-pair failure")
        category_set = frozenset(categories)
        if category_set in seen_sets:
            raise ValueError("duplicate category set")
        seen_sets.add(category_set)
        repeat_intervals = [intervals[index] for index in positions]
        if len({item["candidate_id"] for item in intervals}) != 5 or len({item["source_sha256"] for item in intervals}) != 5:
            raise ValueError("source reused within composition")
        if repeat_intervals[0]["candidate_id"] == repeat_intervals[1]["candidate_id"] or repeat_intervals[0]["source_sha256"] == repeat_intervals[1]["source_sha256"]:
            raise ValueError("repeated category must use different clips")
        if [item["sequence_index"] for item in intervals] != list(range(5)):
            raise ValueError("sequence index failure")
        occurrence_seen: Counter = Counter()
        previous_end = -1.0
        for index, item in enumerate(intervals):
            candidate = eligible.get(item["candidate_id"])
            if candidate is None or candidate["category"] != item["category"] or candidate["audio_sha256"] != item["source_sha256"]:
                raise ValueError("source provenance failure")
            if item["category"] == "vacuum_cleaner" and item["window_duration_sec"] < 3.0:
                raise ValueError("vacuum cleaner occurrence shorter than 3 seconds")
            uses[item["candidate_id"]] += 1
            occurrence_seen[item["category"]] += 1
            if item["category_occurrence_index"] != occurrence_seen[item["category"]]:
                raise ValueError("occurrence index failure")
            expected_group = f"{row['sample_id']}:repeat" if item["category"] == repeated else None
            if item["repetition_group_id"] != expected_group or item["event_instance_id"] != f"{row['sample_id']}:e{index + 1}":
                raise ValueError("event identity failure")
            if item["start_sec"] < previous_end:
                raise ValueError("overlapping intervals")
            previous_end = item["end_sec"]
        repeated_roles[repeated] += 1
        audio_path = root / row["audio_file"]
        if digest(audio_path) != row["audio_sha256"] or public_by_id[row["sample_id"]]["audio_sha256"] != row["audio_sha256"]:
            raise ValueError("audio hash failure")
        with wave.open(str(audio_path), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (base.RATE, 1, 2):
                raise ValueError("audio format failure")
            if handle.getnframes() != round(row["duration_sec"] * base.RATE):
                raise ValueError("duration failure")
    if max(uses.values()) > SOURCE_CAP:
        raise ValueError("source cap failure")
    if set(repeated_roles) != {row["category"] for row in sources if row["eligible_for_generation"] and sum(1 for x in sources if x["eligible_for_generation"] and x["category"] == row["category"]) >= 2}:
        raise ValueError("not all repeatable categories represented")
    if max(repeated_roles.values()) - min(repeated_roles.values()) > 1:
        raise ValueError("repeated-role balance failure")


def build() -> Path:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    sources = base.canonical_sources()
    plans = plan_compositions(sources)
    parent_manifest = PARENT_RELEASE / "data" / "composition_manifest.private.json"
    parent_hash = digest(parent_manifest)
    stage = Path(tempfile.mkdtemp(prefix=".repeated-v2-", dir=ROOT))
    try:
        for name in ("audio", "sources", "data"):
            (stage / name).mkdir()
        for source in sources:
            shutil.copy2(PROJECT / source["source_path_local"], stage / "sources" / f"{source['candidate_id']}.wav")
        private = []
        for plan in plans:
            sid_key = json.dumps({"seed": SEED, "pattern": plan["pattern_id"], "ordinal": plan["ordinal"], "sources": [row["candidate_id"] for row in plan["sources"]]}, sort_keys=True)
            sid = "esc50repv2_" + hashlib.sha256(sid_key.encode()).hexdigest()[:14]
            output = np.empty(0, dtype=np.float64)
            intervals = []
            gaps = []
            occurrence_seen: Counter = Counter()
            for position, source in enumerate(plan["sources"]):
                clip, metadata = base.crop(source, 5, plan["ordinal"] * 10 + position)
                if position:
                    gap = round(0.35 + stable("gap", plan["ordinal"], position) % 851 / 1000, 3)
                    output = np.concatenate((output, np.zeros(round(gap * base.RATE))))
                    gaps.append(gap)
                start = len(output)
                output = np.concatenate((output, clip))
                category = source["category"]
                occurrence_seen[category] += 1
                metadata.update({
                    "event_instance_id": f"{sid}:e{position + 1}",
                    "sequence_index": position,
                    "category_occurrence_index": occurrence_seen[category],
                    "repetition_group_id": f"{sid}:repeat" if category == plan["repeat_category"] else None,
                    "start_sec": round(start / base.RATE, 3),
                    "end_sec": round(len(output) / base.RATE, 3),
                })
                intervals.append(metadata)
            target = stage / "audio" / f"{sid}.wav"
            base.write_wav(target, output)
            private.append({
                "schema_version": "esc50-temporal-repeated-v2",
                "sample_id": sid,
                "composition_id_private": f"repeated_{plan['ordinal']:03d}",
                "pattern_id_private": plan["pattern_id"],
                "repeat_category_private": plan["repeat_category"],
                "repeat_positions_private": plan["repeat_positions"],
                "repeat_source_relation_private": "different_approved_clips",
                "scene_mode_private": plan["scene_mode"],
                "scene_private": plan["scene_private"],
                "construction_order": [row["category"] for row in intervals],
                "intervals": intervals,
                "gaps_sec": gaps,
                "duration_sec": round(len(output) / base.RATE, 6),
                "audio_file": f"audio/{sid}.wav",
                "audio_sha256": digest(target),
                "review_state": "unreviewed_composite",
                "release_partition_private": "repeated_extension_42",
                "parent_release_private": "esc50_temporal_scaled_v1",
                "parent_manifest_sha256_private": parent_hash,
                "requires_constraint_review": any(bool(row["source_constraints"]) for row in intervals),
            })
        private.sort(key=lambda row: row["sample_id"])
        public = [{key: row[key] for key in ("sample_id", "duration_sec", "audio_file", "audio_sha256")} for row in private]
        uses = Counter(item["candidate_id"] for row in private for item in row["intervals"])
        repeated_roles = Counter(row["repeat_category_private"] for row in private)
        (stage / "data" / "source_manifest.private.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "data" / "composition_manifest.private.json").write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "data" / "benchmark_manifest.public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {
            "schema_version": "esc50-temporal-repeated-v2",
            "new_files": 42,
            "events_per_file": 5,
            "parent_release": "esc50_temporal_scaled_v1",
            "parent_manifest_path": "../esc50_temporal_scaled_v1/release/data/composition_manifest.private.json",
            "parent_manifest_sha256": parent_hash,
            "pattern_distribution": dict(Counter(row["pattern_id_private"] for row in private)),
            "scene_mode_distribution": dict(Counter(row["scene_mode_private"] for row in private)),
            "repeat_source_relation_distribution": dict(Counter(row["repeat_source_relation_private"] for row in private)),
            "repeated_role_distribution": dict(sorted(repeated_roles.items())),
            "source_use_max": max(uses.values()),
            "source_use_distribution": dict(Counter(uses.values())),
            "forbidden_pairs": len(BANNED_PAIRS),
            "policy": {
                "source_cap": SOURCE_CAP,
                "same_clip_replay_in_main_benchmark": False,
                "nonadjacent_repetitions": True,
                "new_compositions_require_review": True,
                "parent_release_modified": False,
            },
        }
        (stage / "data" / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validate_release(stage)
        stage.replace(OUT)
        return OUT
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        validate_release(OUT)
        print("PASS: 42 repeated-event compositions, approved sources only")
    else:
        print(build())
