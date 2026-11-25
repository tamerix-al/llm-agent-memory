import json
import random
from pathlib import Path
from collections import defaultdict, Counter


FINAL_DATASET_PATH = Path("01_dataset_final.jsonl")
BASE_FACTS_PATH = Path("data/processed/base_facts_upd.jsonl")

OUTPUT_DIR = Path("data/multifact_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "multifact_all_user_facts_full.jsonl"
REPORT_PATH = OUTPUT_DIR / "multifact_all_user_facts_full_report.json"

RANDOM_SEED = 42


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_dialogue_id(row):
    return (
        row.get("dialogue_id")
        or row.get("dialog_id")
        or row.get("dialoug_id")
        or row.get("dialogueId")
        or row.get("dialog_id")
    )


def get_speaker(row):
    return row.get("speaker") or row.get("persona") or row.get("role")


def get_fact_id(row):
    return row.get("fact_id") or row.get("base_fact_id") or row.get("id")


def main():
    random.seed(RANDOM_SEED)

    final_rows = load_jsonl(FINAL_DATASET_PATH)
    base_fact_rows = load_jsonl(BASE_FACTS_PATH)

    # 1. Build fact_id -> base fact metadata
    fact_id_to_record = {}

    for row in base_fact_rows:
        fact_id = get_fact_id(row)
        if not fact_id:
            continue

        fact_id_to_record[str(fact_id)] = row

    # 2. Build user_key -> all facts for that user
    user_to_facts = defaultdict(list)

    for row in base_fact_rows:
        fact_id = get_fact_id(row)
        dialogue_id = get_dialogue_id(row)
        speaker = get_speaker(row)
        fact = row.get("base_fact")

        if not fact_id or not dialogue_id or not speaker or not fact:
            continue

        user_key = f"{dialogue_id}||{speaker}"

        user_to_facts[user_key].append({
            "fact_id": str(fact_id),
            "fact": fact,
            "dialogue_id": dialogue_id,
            "speaker": speaker,
        })

    output_rows = []

    missing_fact_id_metadata = 0
    missing_user_facts = 0
    single_fact_contexts = 0

    for row in final_rows:
        fact_id = str(row.get("fact_id"))

        base_record = fact_id_to_record.get(fact_id)

        if not base_record:
            missing_fact_id_metadata += 1
            dialogue_id = None
            speaker = None
            user_key = f"missing_metadata||{fact_id}"
            memory_facts = [{
                "fact_id": fact_id,
                "fact": row.get("base_fact"),
                "dialogue_id": None,
                "speaker": None,
            }]
        else:
            dialogue_id = get_dialogue_id(base_record)
            speaker = get_speaker(base_record)
            user_key = f"{dialogue_id}||{speaker}"

            memory_facts = list(user_to_facts.get(user_key, []))

            if not memory_facts:
                missing_user_facts += 1
                memory_facts = [{
                    "fact_id": fact_id,
                    "fact": row.get("base_fact"),
                    "dialogue_id": dialogue_id,
                    "speaker": speaker,
                }]

        # Deterministic shuffle so the target fact is not always first
        rng = random.Random(str(row.get("example_id")) + str(RANDOM_SEED))
        rng.shuffle(memory_facts)

        if len(memory_facts) == 1:
            single_fact_contexts += 1

        out = {
            "example_id": row.get("example_id"),
            "fact_id": fact_id,
            "base_fact": row.get("base_fact"),
            "utterance": row.get("utterance"),
            "gold_operation": row.get("target_operation"),
            "tags": row.get("tags"),
            "context_type": "all_user_facts_full",
            "num_memory_facts": len(memory_facts),
            "memory_facts": memory_facts,
            "dialogue_id": dialogue_id,
            "speaker": speaker,
            "user_key": user_key,
        }

        output_rows.append(out)

    save_jsonl(output_rows, OUTPUT_PATH)

    class_counts = Counter(row["gold_operation"] for row in output_rows)
    memory_fact_count_distribution = Counter(row["num_memory_facts"] for row in output_rows)
    user_counts = Counter(row["user_key"] for row in output_rows)

    report = {
        "input_final_dataset": str(FINAL_DATASET_PATH),
        "input_base_facts": str(BASE_FACTS_PATH),
        "output_path": str(OUTPUT_PATH),
        "total_input_examples": len(final_rows),
        "saved_examples": len(output_rows),
        "base_fact_records_loaded": len(base_fact_rows),
        "fact_id_metadata_found": len(fact_id_to_record),
        "user_groups_found": len(user_to_facts),
        "missing_fact_id_metadata_examples": missing_fact_id_metadata,
        "missing_user_facts_examples": missing_user_facts,
        "single_fact_context_examples": single_fact_contexts,
        "class_counts": dict(sorted(class_counts.items())),
        "memory_fact_count_distribution": dict(sorted(memory_fact_count_distribution.items())),
        "top_10_user_group_sizes": user_counts.most_common(10),
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print()
    print(f"Saved dataset to: {OUTPUT_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()