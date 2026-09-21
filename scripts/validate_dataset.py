#!/usr/bin/env python3
"""Standalone integrity checks for the portable benchmark release."""
from __future__ import annotations
import hashlib, json, wave
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; BENCHMARK=ROOT/"benchmark"; DATA=BENCHMARK/"data"
def load(name: str): return json.loads((DATA/name).read_text(encoding="utf-8"))
def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): value.update(block)
    return value.hexdigest()
def has_private_key(value) -> bool:
    if isinstance(value,dict): return any(key.endswith("_private") or has_private_key(child) for key,child in value.items())
    if isinstance(value,list): return any(has_private_key(child) for child in value)
    return False

def main() -> None:
    comps=load("composition_manifest.private.json"); public_comps=load("benchmark_manifest.public.json")
    private=load("qa.private.json"); public=load("qa.public.json"); report=load("release_report.json")
    by_sample={row["sample_id"]:row for row in comps}; public_comp_ids={row["sample_id"] for row in public_comps}
    assert len(comps)==len(by_sample)==len(public_comps)==112 and set(by_sample)==public_comp_ids
    assert len(private)==len(public)==253 and {row["qa_id"] for row in private}=={row["qa_id"] for row in public}
    for row in comps:
        path=BENCHMARK/row["audio_file"]
        assert path.is_file() and digest(path)==row["audio_sha256"]
        assert not {"engine","vacuum_cleaner"}<={event["category"] for event in row["intervals"]}
        with wave.open(str(path),"rb") as handle:
            assert (handle.getnchannels(),handle.getsampwidth(),handle.getframerate())==(1,2,44100)
    public_by_id={row["qa_id"]:row for row in public}; per_audio=Counter(); positions=Counter()
    for row in private:
        assert row["sample_id"] in by_sample and len(row["options"])==4
        assert [option["key"] for option in row["options"]]==list("ABCD")
        assert len({option["text"] for option in row["options"]})==4 and row["answer_key_private"] in "ABCD"
        assert "longest_duration" not in row.get("question_family_private","")
        public_row=public_by_id[row["qa_id"]]; assert not has_private_key(public_row)
        assert public_row["options"]==[{"key":option["key"],"text":option["text"]} for option in row["options"]]
        per_audio[row["sample_id"]]+=1; positions[row["answer_key_private"]]+=1
    assert min(per_audio.values())>=1 and max(per_audio.values())<=3
    assert max(positions.values())-min(positions.values())<=1
    assert report["audio_files"]==112 and report["qa_items"]==253
    print("PASS: 112 audio files, 253 QA items")

if __name__=="__main__": main()
