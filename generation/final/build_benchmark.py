#!/usr/bin/env python3
"""Assemble the release-ready temporal QA benchmark from validated parents."""
from __future__ import annotations
import argparse, hashlib, json, shutil, tempfile, wave
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent; TC=ROOT.parent; OUT=ROOT/"release"
SCALED=TC/"esc50_temporal_scaled_v1"/"release"
REPEATED=TC/"esc50_temporal_repeated_v2"/"release"
PARENTS=(
    ("scaled_v1",SCALED),
    ("repeated_v2",REPEATED),
)

def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): value.update(block)
    return value.hexdigest()
def has_private_key(value) -> bool:
    if isinstance(value,dict): return any(key.endswith("_private") or has_private_key(child) for key,child in value.items())
    if isinstance(value,list): return any(has_private_key(child) for child in value)
    return False

def validate(root: Path) -> None:
    comps=load(root/"data"/"composition_manifest.private.json"); public_comps=load(root/"data"/"benchmark_manifest.public.json")
    private=load(root/"data"/"qa.private.json"); public=load(root/"data"/"qa.public.json"); report=load(root/"data"/"release_report.json")
    by_sample={row["sample_id"]:row for row in comps}; public_by_sample={row["sample_id"]:row for row in public_comps}
    if len(comps)!=112 or len(by_sample)!=112 or len(public_comps)!=112: raise ValueError("composition count failure")
    if len(private)!=253 or len(public)!=253: raise ValueError("QA count failure")
    if {row["qa_id"] for row in private}!={row["qa_id"] for row in public}: raise ValueError("QA public/private mismatch")
    public_by_id={row["qa_id"]:row for row in public}; per_audio=Counter(); positions=Counter()
    for row in comps:
        categories={event["category"] for event in row["intervals"]}
        if {"engine","vacuum_cleaner"}<=categories: raise ValueError("forbidden engine/vacuum composition")
        target=root/row["audio_file"]
        if digest(target)!=row["audio_sha256"] or public_by_sample[row["sample_id"]]["audio_sha256"]!=row["audio_sha256"]: raise ValueError("audio hash failure")
        with wave.open(str(target),"rb") as handle:
            if handle.getnchannels()!=1 or handle.getsampwidth()!=2 or handle.getframerate()!=44100: raise ValueError("audio format failure")
    for row in private:
        if row["sample_id"] not in by_sample or row["audio_file"]!=by_sample[row["sample_id"]]["audio_file"]: raise ValueError("QA audio reference failure")
        if len(row["options"])!=4 or [option["key"] for option in row["options"]]!=list("ABCD") or len({option["text"] for option in row["options"]})!=4: raise ValueError("four-option failure")
        if row["answer_key_private"] not in "ABCD": raise ValueError("answer key failure")
        if "longest_duration" in row.get("question_family_private","") or "longest duration" in row["question"].lower(): raise ValueError("duration question survived")
        if row.get("release_status_private")!="ready_by_user_instruction": raise ValueError("release gate failure")
        public_row=public_by_id[row["qa_id"]]
        if has_private_key(public_row) or public_row["options"]!=[{"key":option["key"],"text":option["text"]} for option in row["options"]]: raise ValueError("public leakage/projection failure")
        per_audio[row["sample_id"]]+=1; positions[row["answer_key_private"]]+=1
    if min(per_audio.values())<1 or max(per_audio.values())>3: raise ValueError("per-audio QA cap failure")
    if max(positions.values())-min(positions.values())>1: raise ValueError("answer-position imbalance")
    if report["audio_files"]!=112 or report["qa_items"]!=253: raise ValueError("report mismatch")

def build() -> Path:
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    stage=Path(tempfile.mkdtemp(prefix=".benchmark-v1-",dir=ROOT))
    try:
        (stage/"audio").mkdir(); (stage/"data").mkdir(); private_q=[]; compositions=[]
        parent_hashes={}
        for release_name,parent in PARENTS:
            source_q=load(parent/"data"/("qa_drafts.private.json"))
            source_comps=load(parent/"data"/"composition_manifest.private.json")
            parent_hashes[release_name]={
                "composition_manifest_sha256":digest(parent/"data"/"composition_manifest.private.json"),
                "qa_private_sha256":digest(parent/"data"/"qa_drafts.private.json"),
            }
            selected={row["sample_id"] for row in source_q}; comp_by_id={row["sample_id"]:row for row in source_comps}
            for sample_id in sorted(selected):
                source=comp_by_id[sample_id]; copied=dict(source); copied["source_release_private"]=release_name; copied["audio_file"]=f"audio/{sample_id}.wav"
                shutil.copy2(parent/source["audio_file"],stage/copied["audio_file"]); compositions.append(copied)
            for source in source_q:
                row=dict(source); row["source_release_private"]=release_name; row["source_draft_status_private"]=row.pop("draft_status",None); row["release_status_private"]="ready_by_user_instruction"; row["audio_file"]=f"audio/{row['sample_id']}.wav"; private_q.append(row)
        compositions.sort(key=lambda row:row["sample_id"]); private_q.sort(key=lambda row:row["qa_id"])
        public_comps=[{key:row[key] for key in ("sample_id","duration_sec","audio_file","audio_sha256")} for row in compositions]
        public_q=[{**{key:row[key] for key in ("qa_id","sample_id","audio_file","question")},"options":[{"key":option["key"],"text":option["text"]} for option in row["options"]]} for row in private_q]
        per_audio=Counter(row["sample_id"] for row in private_q)
        report={
            "schema_version":"esc50-temporal-benchmark-v1","audio_files":len(compositions),"qa_items":len(private_q),
            "audio_by_source_release":dict(Counter(row["source_release_private"] for row in compositions)),
            "qa_by_source_release":dict(Counter(row["source_release_private"] for row in private_q)),
            "qa_by_family":dict(sorted(Counter(row["question_family_private"] for row in private_q).items())),
            "questions_per_audio":dict(sorted(Counter(per_audio.values()).items())),
            "answer_positions":dict(sorted(Counter(row["answer_key_private"] for row in private_q).items())),
            "parent_hashes":parent_hashes,
            "policy":{"all_items_four_option":True,"max_questions_per_audio":3,"longest_duration_removed":True,"engine_vacuum_cooccurrence_removed":True,"release_gate":"ready_by_user_instruction"},
        }
        (stage/"data"/"composition_manifest.private.json").write_text(json.dumps(compositions,ensure_ascii=False,indent=2)+"\n")
        (stage/"data"/"benchmark_manifest.public.json").write_text(json.dumps(public_comps,ensure_ascii=False,indent=2)+"\n")
        (stage/"data"/"qa.private.json").write_text(json.dumps(private_q,ensure_ascii=False,indent=2)+"\n")
        (stage/"data"/"qa.public.json").write_text(json.dumps(public_q,ensure_ascii=False,indent=2)+"\n")
        (stage/"data"/"release_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        validate(stage); stage.replace(OUT); return OUT
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--validate",action="store_true"); args=parser.parse_args()
    if args.validate: validate(OUT); print("PASS: release-ready 112 audio / 253 QA benchmark")
    else: print(build())
