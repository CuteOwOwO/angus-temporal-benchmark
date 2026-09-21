#!/usr/bin/env python3
"""Generate diverse, deterministic temporal QA drafts for scaled v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "release" / "data"
PRIVATE = DATA / "composition_manifest.private.json"

DISPLAY = {
    "breathing":"breathing", "brushing_teeth":"tooth brushing", "car_horn":"a car horn",
    "cat":"a cat", "chainsaw":"a chainsaw", "chirping_birds":"birds chirping",
    "church_bells":"bells", "clapping":"clapping", "clock_alarm":"an alarm clock",
    "clock_tick":"a ticking clock", "coughing":"coughing", "cow":"a cow",
    "crickets":"crickets", "crow":"a crow", "dog":"a dog",
    "door_wood_creaks":"an opening or creaking door", "door_wood_knock":"a knock on a wooden door",
    "engine":"an engine", "fireworks":"fireworks", "footsteps":"footsteps", "frog":"a frog",
    "glass_breaking":"breaking glass", "insects":"insects", "keyboard_typing":"keyboard typing",
    "laughing":"laughter", "mouse_click":"a mouse click", "pouring_water":"water pouring",
    "rain":"rain", "rooster":"a rooster", "sea_waves":"sea waves", "siren":"a siren",
    "sneezing":"sneezing", "snoring":"snoring", "thunderstorm":"a thunderstorm",
    "toilet_flush":"a toilet flush", "vacuum_cleaner":"a vacuum cleaner", "water_drops":"water drops",
}

EXCLUDED = {
    "esc50div13_9006eabfde1dc6":{"footsteps"}, "esc50div13_9469b12acf8602":{"mouse_click"},
    "esc50div13_bdfd9c9be81260":{"door_wood_knock"}, "esc50div13_c5ec0d236aa412":{"laughing"},
    "esc50scaled75_9bb07941cd5e31":{"thunderstorm"},
}
BLOCKED = {
    "esc50div13_25efb61a0206b3":"fireworks was not distinct enough in human review",
    "esc50div13_cafd1ab9cea73d":"frog was not distinct enough in human review",
    "esc50scaled75_a851f80bdd945a":"engine and vacuum cleaner were not perceptually distinguishable",
}

ACOUSTIC_GROUPS = (
    {"breathing","coughing","laughing","sneezing","snoring"},
    {"engine","vacuum_cleaner","chainsaw"},
    {"siren","car_horn","clock_alarm","church_bells"},
    {"rain","sea_waves","pouring_water","water_drops","toilet_flush","brushing_teeth"},
    {"crickets","insects","chirping_birds","frog"},
    {"crow","rooster","cow","cat","dog"},
    {"clapping","glass_breaking","door_wood_knock","mouse_click","keyboard_typing","clock_tick"},
    {"door_wood_creaks","footsteps"},
)
GROUP_BY_CATEGORY={category:index for index,group in enumerate(ACOUSTIC_GROUPS) for category in group}


def stable(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "big")


def qid(sample_id: str, family: str, ordinal: int) -> str:
    return "qas75_" + hashlib.sha256(f"{sample_id}:{family}:{ordinal}".encode()).hexdigest()[:14]


def shuffled(values: list, key: str) -> list:
    return sorted(values, key=lambda value: hashlib.sha256(f"{key}:{value}".encode()).hexdigest())


def confusable(left: str, right: str) -> bool:
    group=GROUP_BY_CATEGORY.get(left)
    return left!=right and group is not None and group==GROUP_BY_CATEGORY.get(right)


def event_options(categories: list[str], answer: str, key: str, context: list[str] | None = None, forbidden: set[str] | None = None) -> list[dict]:
    context=list(dict.fromkeys(context or categories)); forbidden=set(forbidden or ())
    distractors=[]
    for value in shuffled(list(dict.fromkeys(categories)),key+":present"):
        if value==answer or value in forbidden or confusable(value,answer) or any(confusable(value,old) for old in distractors): continue
        distractors.append(value)
        if len(distractors)==3: break
    fillers=[value for value in DISPLAY if value not in context and value!=answer and value not in forbidden]
    strict=[value for value in fillers if not any(confusable(value,old) for old in context+distractors+[answer])]
    relaxed=[value for value in fillers if not confusable(value,answer) and not any(confusable(value,old) for old in distractors)]
    for value in shuffled(strict,key+":strict")+shuffled(relaxed,key+":relaxed")+shuffled(fillers,key+":fallback"):
        if len(distractors)>=3: break
        if value in distractors or any(confusable(value,old) for old in distractors+[answer]): continue
        distractors.append(value)
    if len(distractors)!=3: raise ValueError(f"cannot build four distinct options for {answer}")
    choices = distractors + [answer]
    choices = shuffled(choices, key + ":position")
    return [{"key": chr(65+i), "text": DISPLAY[value], "value_private": value} for i, value in enumerate(choices)]


def numeric_options(maximum: int, answer: int, key: str) -> list[dict]:
    pool=list(range(max(3,maximum,answer+2)+1))
    alternatives=sorted((value for value in pool if value!=answer),key=lambda value:(abs(value-answer),stable(key,"numeric",value)))
    values=[answer]+alternatives[:3]
    values = shuffled(values, key + ":numeric")
    return [{"key": chr(65+i), "text": str(value), "value_private": value} for i, value in enumerate(values)]


def sequence_options(order: list[str], key: str) -> list[dict]:
    variants = [tuple(order), tuple(reversed(order))]
    for index in range(len(order)-1):
        value=list(order); value[index],value[index+1]=value[index+1],value[index]; variants.append(tuple(value))
    variants.append(tuple(order[1:]+order[:1]))
    unique=[]
    for value in variants:
        if value not in unique: unique.append(value)
    answer=tuple(order); distractors=shuffled([value for value in unique if value!=answer],key)[:3]
    choices=shuffled(distractors+[answer],key+":position")
    return [{"key":chr(65+i),"text":" → ".join(DISPLAY[value] for value in choice),"value_private":list(choice)} for i,choice in enumerate(choices)]


def add(records: list[dict], sample: dict, family: str, ordinal: int, question: str, options: list[dict], answer, evidence: dict, status: str, perceptual: bool = False) -> None:
    key=qid(sample["sample_id"],family,ordinal)
    answer_index=next(index for index,option in enumerate(options) if option["value_private"]==answer)
    required_reviews=[]
    if status=="pending_composition_review": required_reviews.append("composition")
    if perceptual: required_reviews.append("perceptual")
    if required_reviews==["composition","perceptual"]: draft_status="pending_composition_and_perceptual_review"
    elif required_reviews==["perceptual"]: draft_status="pending_perceptual_review"
    else: draft_status=status
    records.append({
        "qa_id":key,"sample_id":sample["sample_id"],"audio_file":sample["audio_file"],
        "draft_status":draft_status,
        "question_family_private":family,"question":question,
        "options":[{"key":option["key"],"text":option["text"]} for option in options],
        "answer_key_private":chr(65+answer_index),"answer_value_private":answer,
        "evidence_private":evidence,"required_reviews_private":required_reviews,
        "perceptual_review_required_private":perceptual,
    })


def standard_status(sample: dict) -> str:
    return "ready_from_review" if sample.get("review_state","").startswith("reviewed_pass") else "pending_composition_review"


def questions_for(sample: dict) -> list[dict]:
    order=[row["category"] for row in sample["intervals"]]
    excluded=EXCLUDED.get(sample["sample_id"],set()); safe=[value for value in order if value not in excluded]
    if len(safe)<2: return []
    status=standard_status(sample); records=[]; ordinal=0; n=len(order)

    if n>=3 and not excluded:
        ordinal+=1; key=qid(sample["sample_id"],"sequence_selection",ordinal); options=sequence_options(order,key)
        add(records,sample,"sequence_selection",ordinal,"Which sequence matches the order in which the sound types begin?",options,order,{"event_indices":list(range(n))},status)

    # Generate one candidate for each robust onset-order family. A global
    # scheduler below selects only two or three questions per recording.
    safe_indices=[index for index,value in enumerate(order) if value in safe]
    pairs=[(i,j) for i in safe_indices for j in safe_indices if i<j]
    i,j=pairs[stable(sample["sample_id"],"pair")%len(pairs)]; answer=order[i]
    ordinal+=1; key=qid(sample["sample_id"],"pair_order",ordinal); options=event_options([order[i],order[j]],answer,key,context=order)
    add(records,sample,"pair_order",ordinal,"Which of these sound types begins earlier?",options,answer,{"event_indices":[i,j]},status)

    if order[0] in safe:
        ordinal+=1; answer=order[0]; key=qid(sample["sample_id"],"first",ordinal); options=event_options(safe,answer,key,context=order)
        add(records,sample,"first",ordinal,"Which of these sound types begins first?",options,answer,{"event_indices":[0]},status)

    if order[-1] in safe:
        ordinal+=1; answer=order[-1]; key=qid(sample["sample_id"],"last",ordinal); options=event_options(safe,answer,key,context=order)
        add(records,sample,"last",ordinal,"Which of these sound types begins last?",options,answer,{"event_indices":[n-1]},status)

    adjacent=[(i,i+1) for i in range(n-1) if order[i] in safe and order[i+1] in safe]
    if adjacent:
        i,j=adjacent[stable(sample["sample_id"],"adjacent")%len(adjacent)]; answer=order[j]
        ordinal+=1; key=qid(sample["sample_id"],"next_event",ordinal); options=event_options(safe,answer,key,context=order,forbidden={order[i]})
        add(records,sample,"next_event",ordinal,f"Which listed sound type begins next after {DISPLAY[order[i]]}?",options,answer,{"event_indices":[i,j]},status)

    if n>=3 and not excluded:
        positions=[index for index in range(1,n-1) if order[index] in safe]
        if positions:
            index=positions[stable(sample["sample_id"],"ordinal")%len(positions)]; label={1:"second",2:"third"}.get(index,f"position {index+1}")
            ordinal+=1; answer=order[index]; key=qid(sample["sample_id"],"ordinal_position",ordinal); options=event_options(safe,answer,key,context=order)
            add(records,sample,"ordinal_position",ordinal,f"Which sound type begins {label}?",options,answer,{"event_indices":[index],"ordinal":index+1},status)

    nonlocal_pairs=[(i,j) for i in range(n) for j in range(i+2,n) if order[i] in safe and order[j] in safe]
    if nonlocal_pairs:
        i,j=nonlocal_pairs[stable(sample["sample_id"],"nonlocal")%len(nonlocal_pairs)]; ask_later=stable(sample["sample_id"],"later")%2==1; answer=order[j] if ask_later else order[i]
        ordinal+=1; key=qid(sample["sample_id"],"nonlocal_pair_order",ordinal); options=event_options([order[i],order[j]],answer,key,context=order)
        word="later" if ask_later else "earlier"; add(records,sample,"nonlocal_pair_order",ordinal,f"Which sound type begins {word}, {DISPLAY[order[i]]} or {DISPLAY[order[j]]}?",options,answer,{"event_indices":[i,j]},status)

    anchors=[i for i in range(1,n-1) if not excluded and order[i] in safe]
    if anchors:
        i=anchors[stable(sample["sample_id"],"count")%len(anchors)]; count_after=stable(sample["sample_id"],"count-side")%2==1; answer=n-i-1 if count_after else i
        ordinal+=1; key=qid(sample["sample_id"],"count_before_after",ordinal); options=numeric_options(n-1,answer,key); side="after" if count_after else "before"
        add(records,sample,"count_before_after",ordinal,f"How many different sound types begin {side} {DISPLAY[order[i]]}?",options,answer,{"anchor_index":i,"count":answer},status)

    between_pairs=[(i,j) for i in range(n) for j in range(i+2,n) if order[i] in safe and order[j] in safe]
    if n>=4 and between_pairs and not excluded:
        i,j=between_pairs[stable(sample["sample_id"],"between")%len(between_pairs)]; answer=j-i-1
        ordinal+=1; key=qid(sample["sample_id"],"count_between",ordinal); options=numeric_options(n-2,answer,key)
        add(records,sample,"count_between",ordinal,f"How many different sound types begin between {DISPLAY[order[i]]} and {DISPLAY[order[j]]}?",options,answer,{"event_indices":[i,j],"count":answer},status)

    # Gap questions concern the deliberately inserted inter-segment pauses and
    # remain gated by perceptual review.
    if n>=3 and sample.get("gaps_sec") and not excluded:
        gaps=sample["gaps_sec"]; ranked=sorted(range(len(gaps)),key=lambda i:gaps[i]); best,second=ranked[-1],ranked[-2]
        if gaps[best]-gaps[second]>=0.3 and stable(sample["sample_id"],"gap")%4==0:
            answer=order[best]; ordinal+=1; key=qid(sample["sample_id"],"longest_gap_after",ordinal); option_values=order[:-1]; options=event_options(option_values,answer,key,context=order)
            add(records,sample,"longest_gap_after",ordinal,"Which labeled sound type is followed by the longest pause before the next sound type begins?",options,answer,{"gap_indices":list(range(len(gaps))),"gaps_sec":gaps,"margin_sec":round(gaps[best]-gaps[second],3)},status,True)
    return records


CORE_FAMILIES = ("pair_order", "first", "last", "next_event")
SECONDARY_TARGETS = {
    "longest_gap_after": 4,
    "count_between": 4,
    "count_before_after": 4,
    "sequence_selection": 4,
    "ordinal_position": 4,
}


def core_compatible(candidate: dict, chosen: list[dict], event_count: int) -> bool:
    family=candidate["question_family_private"]; indices=tuple(candidate["evidence_private"]["event_indices"])
    for prior in chosen:
        other=prior["question_family_private"]
        if other==family or tuple(prior["evidence_private"]["event_indices"])==indices: return False
        pair=indices if family=="pair_order" else tuple(prior["evidence_private"]["event_indices"]) if other=="pair_order" else None
        next_pair=indices if family=="next_event" else tuple(prior["evidence_private"]["event_indices"]) if other=="next_event" else None
        families={family,other}
        if families=={"pair_order","first"} and 0 in pair: return False
        if families=={"pair_order","last"} and event_count-1 in pair: return False
        if families=={"pair_order","next_event"} and pair==next_pair: return False
        if families=={"first","next_event"} and next_pair[0]==0: return False
        if families=={"last","next_event"} and next_pair[1]==event_count-1: return False
    return True


def schedule_questions(samples: list[dict], candidates: list[dict]) -> list[dict]:
    by_sample={sample["sample_id"]:[] for sample in samples}
    sample_by_id={sample["sample_id"]:sample for sample in samples}
    for row in candidates: by_sample[row["sample_id"]].append(row)
    selected={sample_id:[] for sample_id in by_sample}; core_counts=Counter()
    ordered=sorted(samples,key=lambda sample:(stable(sample["sample_id"],"core-order"),sample["sample_id"]))
    for sample in ordered:
        sample_id=sample["sample_id"]; event_count=len(sample["intervals"]); wanted=1 if event_count==2 else 2
        core=[row for row in by_sample[sample_id] if row["question_family_private"] in CORE_FAMILIES]
        valid=[]
        for size in range(min(wanted,len(core)),0,-1):
            valid=[combo for combo in combinations(core,size) if all(core_compatible(row,[other for other in combo if other is not row],event_count) for row in combo)]
            if valid: break
        if not valid: continue
        def combo_score(combo):
            projected=core_counts.copy(); projected.update(row["question_family_private"] for row in combo)
            values=[projected[family] for family in CORE_FAMILIES]
            names=tuple(sorted(row["question_family_private"] for row in combo))
            return max(values)-min(values),sum(value*value for value in values),stable(sample_id,"core-combo",names)
        choice=min(valid,key=combo_score); selected[sample_id].extend(choice); core_counts.update(row["question_family_private"] for row in choice)

    # Add exactly 20 diverse third questions, balanced across five secondary
    # families. Each comes from a different recording.
    used_for_secondary=set()
    for family,target in SECONDARY_TARGETS.items():
        choices=[]
        for sample in samples:
            sample_id=sample["sample_id"]
            if sample_id in used_for_secondary or len(selected[sample_id])>=3: continue
            row=next((value for value in by_sample[sample_id] if value["question_family_private"]==family),None)
            if row: choices.append(row)
        choices.sort(key=lambda row:(stable(row["sample_id"],"secondary",family),row["sample_id"]))
        if len(choices)<target: raise ValueError(f"not enough candidates for {family}: {len(choices)}")
        for row in choices[:target]:
            selected[row["sample_id"]].append(row); used_for_secondary.add(row["sample_id"])

    result=[row for sample_id in sorted(selected) for row in selected[sample_id]]
    counts=Counter(row["sample_id"] for row in result)
    if not counts or min(counts.values())<1 or max(counts.values())>3: raise ValueError(f"per-sample QA cap failure: {counts}")
    core=Counter(row["question_family_private"] for row in result if row["question_family_private"] in CORE_FAMILIES)
    if max(core.values())-min(core.values())>2: raise ValueError(f"core family imbalance: {core}")
    if any(row["question_family_private"]=="longest_duration" for row in result): raise ValueError("duration question survived scheduling")
    return result


def validate(public: list[dict], private: list[dict]) -> None:
    if len(public)!=len(private) or {row["qa_id"] for row in public}!={row["qa_id"] for row in private}: raise ValueError("public/private mismatch")
    forbidden={"answer_key_private","answer_value_private","evidence_private","question_family_private"}
    if any(forbidden & set(row) for row in public): raise ValueError("answer leakage in public QA")
    for row in private:
        keys=[option["key"] for option in row["options"]]
        texts=[option["text"] for option in row["options"]]
        if len(keys)!=4 or keys!=list("ABCD") or len(texts)!=len(set(texts)) or row["answer_key_private"] not in keys: raise ValueError("invalid four-option item")
        if "as soon as" in row["question"].lower() or "immediately" in row["question"].lower(): raise ValueError("unsupported adjacency wording")


def balance_answer_positions(records: list[dict]) -> None:
    """Balance answer positions separately for 2-, 3-, and 4-option items."""
    counts: dict[int, Counter] = {}
    for row in sorted(records, key=lambda value: value["qa_id"]):
        size = len(row["options"])
        group = counts.setdefault(size, Counter())
        current = ord(row["answer_key_private"]) - 65
        answer = row["options"][current]
        distractors = [option for index, option in enumerate(row["options"]) if index != current]
        target = min(range(size), key=lambda index: (group[index], stable(row["qa_id"], "answer-position", index)))
        reordered = list(distractors)
        reordered.insert(target, answer)
        for index, option in enumerate(reordered):
            option["key"] = chr(65 + index)
        row["options"] = reordered
        row["answer_key_private"] = chr(65 + target)
        group[target] += 1


def main() -> None:
    samples=load(PRIVATE); candidates=[]; blocked=[]; eligible=[]
    for sample in samples:
        categories={row["category"] for row in sample["intervals"]}
        if {"engine","vacuum_cleaner"} <= categories:
            blocked.append({"sample_id":sample["sample_id"],"reason_private":"engine and vacuum cleaner co-occur and were not perceptually distinguishable"}); continue
        if sample["sample_id"] in BLOCKED:
            blocked.append({"sample_id":sample["sample_id"],"reason_private":BLOCKED[sample["sample_id"]]}); continue
        eligible.append(sample); candidates.extend(questions_for(sample))
    private=schedule_questions(eligible,candidates)
    balance_answer_positions(private)
    private.sort(key=lambda row:row["qa_id"])
    public=[{key:row[key] for key in ("qa_id","sample_id","audio_file","draft_status","question","options")} for row in private]
    validate(public,private)
    (DATA/"qa_drafts.private.json").write_text(json.dumps(private,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (DATA/"qa_drafts.public.json").write_text(json.dumps(public,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (DATA/"qa_blocked.private.json").write_text(json.dumps(blocked,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    answer_positions_by_size = {
        str(size): dict(Counter(row["answer_key_private"] for row in private if len(row["options"]) == size))
        for size in sorted({len(row["options"]) for row in private})
    }
    if any(max(values.values())-min(values.values())>1 for values in answer_positions_by_size.values()):
        raise ValueError(f"answer position imbalance: {answer_positions_by_size}")
    per_sample=Counter(row["sample_id"] for row in private)
    report={"schema_version":"temporal-scaled75-qa-v2","qa_drafts":len(private),"blocked_compositions":len(blocked),"by_family":dict(Counter(row["question_family_private"] for row in private)),"by_status":dict(Counter(row["draft_status"] for row in private)),"questions_per_sample":dict(Counter(per_sample.values())),"answer_positions_by_option_count":answer_positions_by_size,"policy":{"max_questions_per_sample":3,"two_event_core_cap":1,"balanced_core_onset_families":list(CORE_FAMILIES),"longest_duration_removed":True,"sequence_distractors_use_same_event_set":True,"numeric_answers_are_deterministic":True,"gap_requires_perceptual_review":True,"answer_positions_balanced_within_option_count":True,"unsupported_immediacy_wording_forbidden":True,"public_has_no_answers_or_evidence":True}}
    (DATA/"qa_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Wrote {len(private)} QA drafts across {len(report['by_family'])} families; blocked {len(blocked)} compositions")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__=="__main__": main()
