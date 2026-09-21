#!/usr/bin/env python3
"""Build a 75-clip temporal dataset from human-approved ESC-50 sources."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import shutil
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TC = PROJECT / "temporal_controlled_v1"
OLD = TC / "esc50_temporal_main_pilot_25" / "release"
V2 = TC / "source_screening_v2"
R20 = TC / "source_screening_remaining20"
OUT = ROOT / "release"
RATE, PEAK_CEILING, SEED, SOURCE_CAP = 44_100, 0.90, 20260921_75, 3

NEW_QUOTA = {2: 10, 3: 15, 4: 15, 5: 10}
OLD_BANNED = {
    frozenset(pair) for pair in (
        ("door_wood_knock", "door_wood_creaks"), ("door_wood_knock", "footsteps"),
        ("door_wood_knock", "dog"), ("door_wood_creaks", "footsteps"),
        ("brushing_teeth", "pouring_water"), ("brushing_teeth", "toilet_flush"),
        ("brushing_teeth", "clock_alarm"), ("pouring_water", "toilet_flush"),
        ("toilet_flush", "door_wood_creaks"), ("toilet_flush", "clock_alarm"),
        ("glass_breaking", "siren"), ("glass_breaking", "car_horn"),
        ("glass_breaking", "clapping"), ("glass_breaking", "fireworks"),
        ("fireworks", "siren"), ("fireworks", "clapping"),
        ("fireworks", "church_bells"), ("clock_alarm", "rooster"),
        ("coughing", "laughing"), ("sneezing", "laughing"),
        ("clapping", "laughing"), ("siren", "car_horn"),
    )
}
NEW_BANNED = {
    frozenset(pair) for pair in (
        ("crickets", "chirping_birds"), ("crow", "chirping_birds"),
        ("insects", "crickets"), ("engine", "airplane"),
        ("engine", "chainsaw"), ("water_drops", "pouring_water"),
    )
}
BANNED_PAIRS = OLD_BANNED | NEW_BANNED

SCENES = {
    "domestic_indoor": {
        "breathing", "brushing_teeth", "cat", "clapping", "clock_alarm", "clock_tick",
        "coughing", "dog", "door_wood_creaks", "door_wood_knock", "footsteps",
        "glass_breaking", "keyboard_typing", "laughing", "mouse_click", "pouring_water",
        "sneezing", "snoring", "toilet_flush", "vacuum_cleaner", "water_drops",
    },
    "outdoor_natural": {
        "chainsaw", "chirping_birds", "cow", "crickets", "crow", "dog", "footsteps",
        "frog", "insects", "rain", "rooster", "sea_waves", "thunderstorm",
    },
    "urban_community": {
        "car_horn", "chainsaw", "church_bells", "clapping", "dog", "engine", "fireworks",
        "footsteps", "glass_breaking", "laughing", "siren", "thunderstorm",
    },
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable(*parts: object) -> int:
    body = json.dumps([SEED, *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:8], "big")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"unexpected WAV format: {path}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    if not np.isfinite(audio).all() or np.max(np.abs(audio), initial=0) > PEAK_CEILING + 1e-8:
        raise ValueError("unsafe output audio")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(RATE)
        handle.writeframes(np.rint(audio * 32767).astype("<i2").tobytes())


def frame_activity(audio: np.ndarray) -> tuple[np.ndarray, float]:
    frame = round(0.02 * RATE)
    framed = audio[: len(audio) // frame * frame].reshape(-1, frame)
    levels = 20 * np.log10(np.maximum(np.sqrt(np.mean(framed ** 2, axis=1)), 1e-12))
    active = levels >= max(-50.0, float(levels.max()) - 35.0)
    return active, float(active.sum() * frame / RATE)


def active_bounds(audio: np.ndarray) -> tuple[int, int, np.ndarray]:
    frame, hop = round(0.02 * RATE), round(0.01 * RATE)
    starts = np.arange(0, max(1, len(audio) - frame + 1), hop)
    levels = 20 * np.log10(np.maximum(np.array([np.sqrt(np.mean(audio[i:i + frame] ** 2)) for i in starts]), 1e-12))
    active = np.flatnonzero(levels >= max(-50.0, float(levels.max()) - 35.0))
    if not len(active):
        raise ValueError("inaudible source")
    mask = np.zeros(len(audio), dtype=bool)
    for index in active:
        mask[starts[index]:min(len(audio), starts[index] + frame)] = True
    start = max(0, int(starts[active[0]]) - round(0.12 * RATE))
    end = min(len(audio), int(starts[active[-1]]) + frame + round(0.12 * RATE))
    return start, end, mask


def dbfs(audio: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(audio ** 2))), 1e-12)))


def canonical_sources() -> list[dict]:
    v2_pool = {row["candidate_id"]: row for row in load(V2 / "data" / "pool.json")["candidates"]}
    v2_notes = load(V2 / "data" / "annotations.json")
    r20_pool = {row["candidate_id"]: row for row in load(R20 / "data" / "pool.json")["candidates"]}
    r20_review = load(R20 / "data" / "review_interpretation.json")["candidates"]
    rows = []
    for candidate_id, review in sorted(v2_notes.items()):
        if review.get("status") != "good":
            continue
        source = V2 / "audio" / f"{candidate_id}.wav"
        constraints = {}
        if candidate_id == "esc50v2_coughing_03": constraints["active_region_review"] = "trim trailing silence"
        if candidate_id in {"esc50v2_door_wood_knock_01", "esc50v2_door_wood_knock_03"}: constraints["gain_review"] = True
        if candidate_id == "esc50v2_footsteps_02": constraints["quality_note"] = "reviewer preferred earlier footsteps batch"
        rows.append({
            "candidate_id": candidate_id, "category": review["category"], "filename": review["filename"],
            "fold": v2_pool[candidate_id]["fold"], "source_batch": "source_screening_v2",
            "source_path_local": str(source.relative_to(PROJECT)), "audio_sha256": digest(source),
            "review_note": review.get("note", ""), "constraints": constraints,
            "eligible_for_generation": candidate_id != "esc50v2_clapping_02",
            "exclusion_reason": "possible clap-and-cheer mixed event" if candidate_id == "esc50v2_clapping_02" else None,
        })
    for review in r20_review:
        if review["verdict"] != "foreground":
            continue
        candidate_id = review["candidate_id"]; pool = r20_pool[candidate_id]
        source = R20 / "audio" / f"{candidate_id}.wav"
        rows.append({
            "candidate_id": candidate_id, "category": review["source"]["category"],
            "filename": review["source"]["filename"], "fold": review["source"]["fold"],
            "source_batch": "source_screening_remaining20", "source_path_local": str(source.relative_to(PROJECT)),
            "audio_sha256": digest(source), "review_note": review["original_note"],
            "constraints": review["constraints"], "eligible_for_generation": True, "exclusion_reason": None,
        })
    if len(rows) != 96 or len({row["candidate_id"] for row in rows}) != 96:
        raise ValueError("canonical source registry must contain exactly 96 unique approved sources")
    return rows


def valid_set(categories: tuple[str, ...], existing_sets: list[frozenset]) -> bool:
    value = frozenset(categories)
    return (
        len(value) == len(categories)
        and not any(pair <= value for pair in BANNED_PAIRS)
        and value not in existing_sets
        and all(len(value & old) <= 2 for old in existing_sets)
    )


def weighted_sample(rng: random.Random, values: list[str], weights: dict[str, int], count: int) -> tuple[str, ...]:
    pool, result = list(values), []
    for _ in range(count):
        total = sum(max(1, weights[value]) for value in pool)
        pick = rng.uniform(0, total); cursor = 0.0
        for value in pool:
            cursor += max(1, weights[value])
            if cursor >= pick:
                result.append(value); pool.remove(value); break
    return tuple(result)


def plan_compositions(sources: list[dict], old_rows: list[dict]) -> list[dict]:
    eligible = [row for row in sources if row["eligible_for_generation"]]
    current = Counter(interval["candidate_id"] for row in old_rows for interval in row["intervals"])
    if not set(current) <= {row["candidate_id"] for row in eligible}:
        raise ValueError("existing release uses a source outside the eligible canonical bank")
    if max(current.values()) > SOURCE_CAP:
        raise ValueError("existing source use already exceeds cap")
    by_category = defaultdict(list)
    for row in eligible: by_category[row["category"]].append(row)
    base_sets = [frozenset(row["construction_order"]) for row in old_rows]
    event_counts = [n for n, count in NEW_QUOTA.items() for _ in range(count)]
    modes = ["coherent_scene"] * 25 + ["cross_context_stress"] * 25

    for attempt in range(500):
        rng = random.Random(SEED + attempt)
        rng.shuffle(event_counts); rng.shuffle(modes)
        remaining = {category: sum(SOURCE_CAP - current[row["candidate_id"]] for row in rows) for category, rows in by_category.items()}
        sets = list(base_sets); plans = []
        failed = False
        for ordinal, (n_events, mode) in enumerate(zip(event_counts, modes), 1):
            found = None
            scene_names = list(SCENES)
            rng.shuffle(scene_names)
            for _ in range(3000):
                if mode == "coherent_scene":
                    scene = scene_names[_ % len(scene_names)]
                    choices = [c for c in SCENES[scene] if remaining.get(c, 0) > 0]
                else:
                    scene = "cross_context_stress"
                    choices = [c for c, capacity in remaining.items() if capacity > 0]
                if len(choices) < n_events: continue
                candidate = weighted_sample(rng, sorted(choices), remaining, n_events)
                if valid_set(candidate, sets):
                    found = candidate, scene; break
            if found is None:
                failed = True; break
            categories, scene = found
            for category in categories: remaining[category] -= 1
            sets.append(frozenset(categories))
            plans.append({"ordinal": ordinal, "scene_mode": mode, "scene_private": scene, "categories": list(categories)})
        if not failed:
            # Assign the least-used eligible source within each selected category.
            assigned_use = Counter(current)
            for plan in plans:
                assigned = []
                for category in plan["categories"]:
                    choices = [row for row in by_category[category] if assigned_use[row["candidate_id"]] < SOURCE_CAP]
                    choices.sort(key=lambda row: (assigned_use[row["candidate_id"]], stable("source", plan["ordinal"], row["candidate_id"])))
                    if not choices: failed = True; break
                    source = choices[0]; assigned_use[source["candidate_id"]] += 1; assigned.append(source)
                plan["sources"] = assigned
                if failed: break
            if not failed and max(assigned_use.values()) <= SOURCE_CAP:
                return plans
    raise RuntimeError("unable to find a valid 50-composition plan")


def crop(source: dict, event_count: int, ordinal: int) -> tuple[np.ndarray, dict]:
    raw = read_wav(PROJECT / source["source_path_local"])
    start_active, end_active, active_mask = active_bounds(raw)
    constraints = source["constraints"]
    if constraints.get("preserve_full_clip"):
        start, clip = 0, raw.copy()
    else:
        trimmed = raw[start_active:end_active]
        lo, hi = {2:(1.2,2.8), 3:(1.1,2.5), 4:(1.0,2.2), 5:(0.9,2.0)}[event_count]
        if source["category"] in {"breathing","footsteps","mouse_click","snoring","water_drops"}: lo, hi = max(lo,1.8), max(hi,3.0)
        if source["category"] == "vacuum_cleaner": lo, hi = max(lo,3.0), max(hi,4.2)
        wanted = lo + stable("duration", source["candidate_id"], ordinal) % 1001 / 1000 * (hi - lo)
        frames = min(len(trimmed), max(round(0.8 * RATE), round(wanted * RATE)))
        if frames < len(trimmed):
            offsets = np.linspace(0, len(trimmed)-frames, num=min(41,len(trimmed)-frames+1), dtype=int)
            start = int(max(offsets, key=lambda value: float(np.mean(trimmed[value:value+frames]**2))))
            clip = trimmed[start:start+frames].copy(); start += start_active
        else:
            start, clip = start_active, trimmed.copy()
    target = -20.0 + (stable("level", source["candidate_id"], ordinal) % 301 / 100 - 1.5)
    gain = min(target - dbfs(raw[active_mask]), 12.0, 20 * math.log10(PEAK_CEILING / max(float(np.max(np.abs(raw))), 1e-12)))
    clip *= 10 ** (gain / 20)
    fade = min(round(0.004 * RATE), len(clip)//8)
    if fade:
        ramp=np.linspace(0,1,fade,endpoint=False); clip[:fade]*=ramp; clip[-fade:]*=ramp[::-1]
    _, active_duration = frame_activity(clip)
    return clip, {
        "candidate_id": source["candidate_id"], "category": source["category"],
        "source_filename": source["filename"], "source_sha256": source["audio_sha256"],
        "window_start_sec": round(start/RATE,3), "window_duration_sec": round(len(clip)/RATE,3),
        "active_duration_sec": round(active_duration,3), "gain_db": round(gain,3),
        "peak": round(float(np.max(np.abs(clip))),5), "source_constraints": constraints,
    }


def validate_release(root: Path) -> None:
    private=load(root/"data"/"composition_manifest.private.json"); public=load(root/"data"/"benchmark_manifest.public.json"); sources=load(root/"data"/"source_manifest.private.json")
    if len(private)!=75 or len(public)!=75 or len(sources)!=96: raise ValueError("release size failure")
    if any(set(row)!={"sample_id","duration_sec","audio_file","audio_sha256"} for row in public): raise ValueError("public metadata leak")
    eligible={row["candidate_id"] for row in sources if row["eligible_for_generation"]}
    uses=Counter(interval["candidate_id"] for row in private for interval in row["intervals"])
    if not set(uses)<=eligible or max(uses.values())>SOURCE_CAP: raise ValueError("source eligibility/use failure")
    sets=[]
    for row in private:
        categories=[interval["category"] for interval in row["intervals"]]; value=frozenset(categories)
        if len(value)!=len(categories) or any(pair<=value for pair in BANNED_PAIRS): raise ValueError("composition category failure")
        if value in sets or any(len(value&old)>2 for old in sets): raise ValueError("composition set overlap failure")
        sets.append(value)
        path=root/row["audio_file"]
        if digest(path)!=row["audio_sha256"]: raise ValueError("audio hash failure")
        with wave.open(str(path),"rb") as handle:
            if (handle.getframerate(),handle.getnchannels(),handle.getsampwidth())!=(RATE,1,2): raise ValueError("audio format failure")
            if handle.getnframes()!=round(row["duration_sec"]*RATE): raise ValueError("duration failure")


def build() -> Path:
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    sources=canonical_sources(); old_rows=load(OLD/"data"/"composition_manifest.private.json"); plans=plan_compositions(sources,old_rows)
    stage=Path(tempfile.mkdtemp(prefix=".scaled75-",dir=ROOT))
    try:
        for name in ("audio","sources","data"): (stage/name).mkdir()
        for source in sources: shutil.copy2(PROJECT/source["source_path_local"],stage/"sources"/f"{source['candidate_id']}.wav")
        private=[]
        for row in old_rows:
            copied=dict(row); copied["release_partition_private"]="inherited_main25"
            shutil.copy2(OLD/row["audio_file"],stage/row["audio_file"]); private.append(copied)
        for plan in plans:
            output=np.empty(0,dtype=np.float64); intervals=[]; gaps=[]
            ordered=sorted(plan["sources"],key=lambda source:stable("order",plan["ordinal"],source["candidate_id"]))
            for position,source in enumerate(ordered):
                clip,meta=crop(source,len(ordered),plan["ordinal"]*10+position)
                if position:
                    gap=round(0.35+(stable("gap",plan["ordinal"],position)%851)/1000,3); output=np.concatenate((output,np.zeros(round(gap*RATE)))); gaps.append(gap)
                start=len(output); output=np.concatenate((output,clip)); meta.update({"start_sec":round(start/RATE,3),"end_sec":round(len(output)/RATE,3)}); intervals.append(meta)
            sample_key = f"{SEED}:{plan['ordinal']}"
            sid = f"esc50scaled75_{hashlib.sha256(sample_key.encode()).hexdigest()[:14]}"
            target = stage / "audio" / f"{sid}.wav"
            write_wav(target, output)
            private.append({
                "sample_id":sid,"composition_id_private":f"scaled_{plan['ordinal']:03d}","scene_mode_private":plan["scene_mode"],"scene_private":plan["scene_private"],
                "construction_order":[row["category"] for row in intervals],"intervals":intervals,"gaps_sec":gaps,
                "duration_sec":round(len(output)/RATE,6),"audio_file":f"audio/{sid}.wav","audio_sha256":digest(target),
                "review_state":"unreviewed_composite","release_partition_private":"scaled_extension_50",
                "requires_constraint_review":any(bool(row["source_constraints"]) for row in intervals),
            })
        private.sort(key=lambda row:row["sample_id"]); public=[{key:row[key] for key in ("sample_id","duration_sec","audio_file","audio_sha256")} for row in private]
        (stage/"data"/"source_manifest.private.json").write_text(json.dumps(sources,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (stage/"data"/"composition_manifest.private.json").write_text(json.dumps(private,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (stage/"data"/"benchmark_manifest.public.json").write_text(json.dumps(public,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        report={
            "schema_version":"esc50-temporal-scaled75-v1","files":len(private),"inherited_files":25,"new_files":50,
            "new_event_count_distribution":dict(Counter(len(row["intervals"]) for row in private if row["release_partition_private"]=="scaled_extension_50")),
            "new_scene_mode_distribution":dict(Counter(row["scene_mode_private"] for row in private if row["release_partition_private"]=="scaled_extension_50")),
            "canonical_sources":96,"eligible_sources":sum(row["eligible_for_generation"] for row in sources),
            "source_use_max":max(Counter(interval["candidate_id"] for row in private for interval in row["intervals"]).values()),
            "forbidden_pairs":len(BANNED_PAIRS),"policy":{"quality_ranked_unreviewed_sources_included":False,"public_manifest_label_free":True,"source_cap":SOURCE_CAP,"new_compositions_require_review":True},
        }
        (stage/"data"/"build_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        validate_release(stage); stage.replace(OUT); return OUT
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--validate",action="store_true"); args=parser.parse_args()
    if args.validate: validate_release(OUT); print("PASS: 75 compositions, approved sources only")
    else: print(build())
