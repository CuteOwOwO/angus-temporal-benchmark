#!/usr/bin/env python3
"""Generate 105 deterministic four-option QA drafts for repeated-event v2."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "release" / "data"
MANIFEST = DATA / "composition_manifest.private.json"
PARENT_QA = ROOT.parent / "esc50_temporal_scaled_v1" / "generate_qa.py"
spec = importlib.util.spec_from_file_location("scaled_v1_qa", PARENT_QA)
parent_qa = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(parent_qa)
DISPLAY = parent_qa.DISPLAY

Q2_FAMILIES = (
    "first_begin", "last_begin", "next_after_first_occurrence",
    "next_after_second_occurrence", "count_between_repetitions",
    "second_occurrence_position",
)
Q3_FAMILIES = ("sequence_selection", "ordinal_position", "occurrence_qualified_pair_order")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def qid(sample_id: str, family: str) -> str:
    return "qarep2_" + hashlib.sha256(f"{sample_id}:{family}".encode()).hexdigest()[:14]


def category_options(order: list[str], answer: str, key: str, required: list[str] | None = None) -> list[dict]:
    choices=[]
    for value in [answer,*(required or [])]:
        if value not in choices: choices.append(value)
    if any(parent_qa.confusable(left,right) for index,left in enumerate(choices) for right in choices[index+1:]):
        raise ValueError("required categorical options are acoustically confusable")
    present=[value for value in dict.fromkeys(order) if value not in choices]
    absent=[value for value in DISPLAY if value not in order and value not in choices]
    for value in parent_qa.shuffled(present,key+":present")+parent_qa.shuffled(absent,key+":absent"):
        if len(choices)>=4: break
        if any(parent_qa.confusable(value,old) for old in choices): continue
        choices.append(value)
    if len(choices)!=4: raise ValueError("could not build four acoustically distinct category options")
    choices=parent_qa.shuffled(choices,key+":position")
    return [{"text":DISPLAY[value],"value_private":value} for value in choices]


def next_category_options(order: list[str], answer: str, anchor: str, key: str) -> list[dict]:
    return parent_qa.event_options(list(dict.fromkeys(order)),answer,key,context=order,forbidden={anchor})


def sequence_options(order: list[str], key: str) -> list[dict]:
    answer = tuple(order)
    variants = []
    candidates = [tuple(reversed(order)), tuple(order[1:] + order[:1])]
    for index in range(4):
        value = list(order)
        value[index], value[index + 1] = value[index + 1], value[index]
        candidates.append(tuple(value))
    for value in sorted(candidates, key=lambda item: stable(key, item)):
        if value != answer and value not in variants:
            variants.append(value)
    choices = [answer, *variants[:3]]
    if len(choices) != 4:
        raise ValueError("could not construct four sequence options")
    return [{"text": " → ".join(DISPLAY[value] for value in choice), "value_private": list(choice)} for choice in choices]


def add(records: list[dict], sample: dict, family: str, question: str, options: list[dict], answer, evidence: dict) -> None:
    matches = [index for index, option in enumerate(options) if option["value_private"] == answer]
    if len(matches) != 1:
        raise ValueError(f"answer is not unique for {sample['sample_id']} {family}")
    for index, option in enumerate(options):
        option["key"] = chr(65 + index)
    records.append({
        "qa_id": qid(sample["sample_id"], family),
        "sample_id": sample["sample_id"],
        "audio_file": sample["audio_file"],
        "draft_status": "pending_composition_review",
        "question_family_private": family,
        "answer_type_private": (
            "sequence" if family == "sequence_selection"
            else "count" if family == "count_between_repetitions"
            else "position" if family == "second_occurrence_position"
            else "category"
        ),
        "question": question,
        "options": options,
        "answer_key_private": chr(65 + matches[0]),
        "answer_value_private": answer,
        "anchor_event_instance_ids_private": evidence.get("anchors", []),
        "evidence_event_instance_ids_private": evidence["events"],
        "derivation_private": evidence["derivation"],
        "required_reviews_private": ["composition"],
    })


def family_legal(sample: dict, family: str) -> bool:
    return family != "next_after_second_occurrence" or sample["repeat_positions_private"][1] < 4


def allocate_q2(samples: list[dict]) -> dict[str, str]:
    scarce = "next_after_second_occurrence"
    eligible_scarce = sorted(
        [sample for sample in samples if family_legal(sample, scarce)],
        key=lambda row: stable("q2-scarce", row["sample_id"]),
    )
    if len(eligible_scarce) < 7:
        raise RuntimeError("not enough samples for next-after-second questions")
    assignments = {sample["sample_id"]: scarce for sample in eligible_scarce[:7]}
    targets = Counter({family: 7 for family in Q2_FAMILIES if family != scarce})
    ordered = sorted(samples, key=lambda row: stable("q2-sample", row["sample_id"]))
    for sample in ordered:
        if sample["sample_id"] in assignments:
            continue
        legal = [family for family in Q2_FAMILIES if targets[family]]
        if not legal:
            raise RuntimeError("unable to allocate balanced Q2 families")
        family = min(legal, key=lambda value: (-targets[value], stable("q2-family", sample["sample_id"], value)))
        assignments[sample["sample_id"]] = family
        targets[family] -= 1
    if any(targets.values()):
        raise RuntimeError(f"incomplete Q2 allocation: {targets}")
    return assignments


def add_family(records: list[dict], sample: dict, family: str) -> None:
    order = sample["construction_order"]
    intervals = sample["intervals"]
    repeated = sample["repeat_category_private"]
    first_repeat, second_repeat = sample["repeat_positions_private"]
    if family == "returning_category":
        category_choices=category_options(order,repeated,qid(sample["sample_id"],family))
        add(records, sample, family, "Which sound type returns later after other sounds have been heard?", category_choices, repeated, {
            "events": [intervals[first_repeat]["event_instance_id"], intervals[second_repeat]["event_instance_id"]],
            "derivation": "the only category assigned to two nonadjacent labeled segments",
        })
    elif family == "first_begin":
        category_choices=category_options(order,order[0],qid(sample["sample_id"],family))
        add(records, sample, family, "Which sound type begins first?", category_choices, order[0], {
            "events": [intervals[0]["event_instance_id"]], "derivation": "minimum sequence_index",
        })
    elif family == "last_begin":
        category_choices=category_options(order,order[-1],qid(sample["sample_id"],family))
        add(records, sample, family, "Which sound type begins last?", category_choices, order[-1], {
            "events": [intervals[-1]["event_instance_id"]], "derivation": "maximum sequence_index",
        })
    elif family in {"next_after_first_occurrence", "next_after_second_occurrence"}:
        anchor = first_repeat if family == "next_after_first_occurrence" else second_repeat
        occurrence = "first" if family == "next_after_first_occurrence" else "second"
        answer_index = anchor + 1
        answer=order[answer_index]
        options=next_category_options(order,answer,repeated,qid(sample["sample_id"],family))
        add(records, sample, family, f"After the {occurrence} section containing {DISPLAY[repeated]}, which sound type begins next?", options, answer, {
            "anchors": [intervals[anchor]["event_instance_id"]],
            "events": [intervals[anchor]["event_instance_id"], intervals[answer_index]["event_instance_id"]],
            "derivation": f"sequence_index {answer_index} follows repeat occurrence {occurrence}",
        })
    elif family == "count_between_repetitions":
        answer = second_repeat - first_repeat - 1
        options = [{"text": str(value), "value_private": value} for value in range(4)]
        add(records, sample, family, f"How many labeled sound sections begin between the two separated sections containing {DISPLAY[repeated]}?", options, answer, {
            "anchors": [intervals[first_repeat]["event_instance_id"], intervals[second_repeat]["event_instance_id"]],
            "events": [item["event_instance_id"] for item in intervals[first_repeat:second_repeat + 1]],
            "derivation": "second repeat index minus first repeat index minus one",
        })
    elif family == "second_occurrence_position":
        answer = second_repeat + 1
        labels = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
        options = [{"text": labels[value], "value_private": value} for value in range(2, 6)]
        add(records, sample, family, f"At which position does {DISPLAY[repeated]} return in the five-part sequence?", options, answer, {
            "anchors": [intervals[second_repeat]["event_instance_id"]],
            "events": [intervals[second_repeat]["event_instance_id"]],
            "derivation": "one-based sequence_index of the second repeated-category segment",
        })
    elif family == "sequence_selection":
        add(records, sample, family, "Which sequence matches the order in which the sound types begin?", sequence_options(order, sample["sample_id"]), order, {
            "events": [item["event_instance_id"] for item in intervals], "derivation": "ascending sequence_index",
        })
    elif family == "ordinal_position":
        index = 1 + stable("ordinal", sample["sample_id"]) % 3
        labels = {1: "second", 2: "third", 3: "fourth"}
        category_choices=category_options(order,order[index],qid(sample["sample_id"],family))
        add(records, sample, family, f"Which sound type begins {labels[index]}?", category_choices, order[index], {
            "events": [intervals[index]["event_instance_id"]], "derivation": f"sequence_index {index}",
        })
    elif family == "occurrence_qualified_pair_order":
        repeat_index = second_repeat
        other_indices = [index for index, category in enumerate(order) if category != repeated and not parent_qa.confusable(category,repeated)]
        if not other_indices: raise ValueError(f"no acoustically distinct pair candidate for {sample['sample_id']}")
        other_index = other_indices[stable("pair", sample["sample_id"]) % len(other_indices)]
        earlier = min(repeat_index, other_index)
        other = order[other_index]
        category_choices=category_options(order,order[earlier],qid(sample["sample_id"],family),required=[repeated,other])
        add(records, sample, family, f"Which sound type begins earlier: the second section containing {DISPLAY[repeated]}, or {DISPLAY[other]}?", category_choices, order[earlier], {
            "anchors": [intervals[repeat_index]["event_instance_id"], intervals[other_index]["event_instance_id"]],
            "events": [intervals[repeat_index]["event_instance_id"], intervals[other_index]["event_instance_id"]],
            "derivation": "smaller sequence_index of the occurrence-qualified pair",
        })
    else:
        raise ValueError(f"unknown family {family}")


def balance_answer_positions(records: list[dict]) -> None:
    counts = Counter()
    for row in sorted(records, key=lambda value: value["qa_id"]):
        current = ord(row["answer_key_private"]) - 65
        answer = row["options"][current]
        distractors = [option for index, option in enumerate(row["options"]) if index != current]
        target = min(range(4), key=lambda index: (counts[index], stable("answer-position", row["qa_id"], index)))
        reordered = list(distractors)
        reordered.insert(target, answer)
        for index, option in enumerate(reordered):
            option["key"] = chr(65 + index)
        row["options"] = reordered
        row["answer_key_private"] = chr(65 + target)
        counts[target] += 1


def validate(private: list[dict], public: list[dict], samples: list[dict]) -> None:
    by_sample = {row["sample_id"]: row for row in samples}
    counts = Counter(row["sample_id"] for row in private)
    if len(private) != 105 or Counter(counts.values()) != Counter({2: 21, 3: 21}):
        raise ValueError("QA/sample quota failure")
    families = Counter(row["question_family_private"] for row in private)
    if families["returning_category"] != 42 or any(families[family] != 7 for family in (*Q2_FAMILIES, *Q3_FAMILIES)):
        raise ValueError(f"family balance failure: {families}")
    if len(public) != len(private) or {row["qa_id"] for row in public} != {row["qa_id"] for row in private}:
        raise ValueError("public/private mismatch")
    def has_private_key(value) -> bool:
        if isinstance(value,dict): return any(key.endswith("_private") or has_private_key(child) for key,child in value.items())
        if isinstance(value,list): return any(has_private_key(child) for child in value)
        return False
    if any(has_private_key(row) for row in public): raise ValueError("private metadata leaked")
    answer_positions = Counter()
    signatures: dict[str, set[tuple]] = {}
    for row in private:
        sample = by_sample[row["sample_id"]]
        options = row["options"]
        if len(options) != 4 or len({item["key"] for item in options}) != 4 or len({json.dumps(item["value_private"], sort_keys=True) for item in options}) != 4:
            raise ValueError("options are not four unique values")
        answer_index = ord(row["answer_key_private"]) - 65
        if options[answer_index]["value_private"] != row["answer_value_private"]:
            raise ValueError("answer key mismatch")
        answer_positions[row["answer_key_private"]] += 1
        family = row["question_family_private"]
        signature = (family, tuple(row["evidence_event_instance_ids_private"]))
        if signature in signatures.setdefault(row["sample_id"], set()):
            raise ValueError("duplicate evidence signature")
        signatures[row["sample_id"]].add(signature)
        question = row["question"].lower()
        if "immediately" in question or "as soon as" in question or "longest duration" in question:
            raise ValueError("ambiguous or forbidden wording")
        order = sample["construction_order"]
        intervals = sample["intervals"]
        positions = sample["repeat_positions_private"]
        evidence_ids = row["evidence_event_instance_ids_private"]
        index_by_id = {item["event_instance_id"]: item["sequence_index"] for item in intervals}
        if family == "returning_category":
            expected = sample["repeat_category_private"]
        elif family == "first_begin":
            expected = order[0]
        elif family == "last_begin":
            expected = order[-1]
        elif family == "next_after_first_occurrence":
            expected = order[positions[0] + 1]
        elif family == "next_after_second_occurrence":
            if positions[1] >= 4:
                raise ValueError("next-after-second question has no following event")
            expected = order[positions[1] + 1]
        elif family == "count_between_repetitions":
            expected = positions[1] - positions[0] - 1
        elif family == "second_occurrence_position":
            expected = positions[1] + 1
        elif family == "sequence_selection":
            expected = order
        elif family == "ordinal_position":
            if len(evidence_ids) != 1 or evidence_ids[0] not in index_by_id:
                raise ValueError("ordinal evidence failure")
            expected = order[index_by_id[evidence_ids[0]]]
        elif family == "occurrence_qualified_pair_order":
            if len(evidence_ids) != 2 or any(value not in index_by_id for value in evidence_ids):
                raise ValueError("pair evidence failure")
            expected = order[min(index_by_id[value] for value in evidence_ids)]
        else:
            raise ValueError(f"unknown family in validator: {family}")
        if row["answer_value_private"] != expected:
            raise ValueError(f"manifest-derived answer failure: {family}")
        if row["answer_type_private"] == "category":
            option_values = {item["value_private"] for item in options}
            if family in {"next_after_first_occurrence","next_after_second_occurrence"}:
                if sample["repeat_category_private"] in option_values or expected not in option_values:
                    raise ValueError("next-event anchor leaked into options")
            if any(parent_qa.confusable(left,right) for left in option_values for right in option_values if left<right):
                raise ValueError("categorical options contain an acoustic-confusion pair")
            if family=="occurrence_qualified_pair_order":
                compared={intervals[index_by_id[value]]["category"] for value in evidence_ids}
                if not compared<=option_values: raise ValueError("pair comparison category missing from options")
    if max(answer_positions.values()) - min(answer_positions.values()) > 1:
        raise ValueError("answer-position imbalance")


def main() -> None:
    samples = load(MANIFEST)
    q2 = allocate_q2(samples)
    q3_samples = sorted(samples, key=lambda row: stable("q3-sample", row["sample_id"]))[:21]
    q3 = {row["sample_id"]: Q3_FAMILIES[index % 3] for index, row in enumerate(q3_samples)}
    records = []
    for sample in samples:
        add_family(records, sample, "returning_category")
        add_family(records, sample, q2[sample["sample_id"]])
        if sample["sample_id"] in q3:
            add_family(records, sample, q3[sample["sample_id"]])
    balance_answer_positions(records)
    records.sort(key=lambda row: row["qa_id"])
    public = [{
        **{key: row[key] for key in ("qa_id", "sample_id", "audio_file", "draft_status", "question")},
        "options":[{"key":option["key"],"text":option["text"]} for option in row["options"]],
    } for row in records]
    validate(records, public, samples)
    (DATA / "qa_drafts.private.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "qa_drafts.public.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": "esc50-temporal-repeated-v2-qa",
        "qa_drafts": len(records),
        "by_family": dict(sorted(Counter(row["question_family_private"] for row in records).items())),
        "questions_per_sample": dict(sorted(Counter(Counter(row["sample_id"] for row in records).values()).items())),
        "answer_positions": dict(sorted(Counter(row["answer_key_private"] for row in records).items())),
        "policy": {"max_qa_per_audio": 3, "all_questions_have_four_options": True, "occurrence_qualified_references": True, "longest_duration_included": False},
    }
    (DATA / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {len(records)} deterministic four-option QA drafts")


if __name__ == "__main__":
    main()
