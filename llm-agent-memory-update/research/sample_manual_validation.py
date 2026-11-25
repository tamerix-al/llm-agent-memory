import csv
import json
import random
from collections import defaultdict
from pathlib import Path


INPUT_PATH = Path("data/final/memory_operations_cleaned_2.jsonl")

OUTPUT_DIR = Path("data/manual_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = OUTPUT_DIR / "manual_validation_sample_100.csv"
OUTPUT_JSONL_PATH = OUTPUT_DIR / "manual_validation_sample_100.jsonl"

RANDOM_SEED = 42

SAMPLE_PLAN = {
    "ADD": 17,
    "DELETE": 17,
    "DELETE_ADD": 17,
    "CONFLICT": 17,
    "IGNORE-NONFACT": 16,
    "IGNORE-OTHERS": 16,
}


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            item["_source_line_number"] = line_number
            records.append(item)

    return records


def get_utterance(item: dict) -> str:
    utterance = item.get("utterance")

    if isinstance(utterance, str):
        return utterance.strip()

    parsed_response = item.get("parsed_response")

    if isinstance(parsed_response, dict):
        utterance = parsed_response.get("utterance")

        if isinstance(utterance, str):
            return utterance.strip()

    return ""


def get_tags_as_text(item: dict) -> str:
    tags = item.get("tags", [])

    if isinstance(tags, list):
        return ";".join(tags)

    return str(tags)


def group_by_operation(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    for item in records:
        operation = item.get("target_operation")

        if operation in SAMPLE_PLAN:
            grouped[operation].append(item)

    return grouped


def make_sample(records: list[dict]) -> list[dict]:
    random.seed(RANDOM_SEED)

    grouped = group_by_operation(records)

    sampled = []

    for operation, n in SAMPLE_PLAN.items():
        items = grouped.get(operation, [])

        if len(items) < n:
            raise ValueError(
                f"Not enough examples for {operation}: need {n}, found {len(items)}"
            )

        selected = random.sample(items, n)
        sampled.extend(selected)

    random.shuffle(sampled)

    return sampled


def flatten_for_annotation(item: dict, annotation_id: int) -> dict:
    base_fact_record = item.get("base_fact_record", {})

    if not isinstance(base_fact_record, dict):
        base_fact_record = {}

    return {
        "annotation_id": annotation_id,
        "example_id": item.get("example_id", ""),
        "fact_id": item.get("fact_id", ""),
        "target_operation": item.get("target_operation", ""),
        "tags": get_tags_as_text(item),
        "base_fact": item.get("base_fact", ""),
        "utterance": get_utterance(item),

        "is_valid": "",
        "error_type": "",
        "comment": "",

        "source_line_number": item.get("_source_line_number", ""),
        "prompt_version": item.get("prompt_version", ""),
        "model": item.get("model", ""),

        "source_dataset": base_fact_record.get("source_dataset", ""),
        "split": base_fact_record.get("split", ""),
        "dialogue_id": base_fact_record.get("dialogue_id", ""),
        "session_id": base_fact_record.get("session_id", ""),
        "speaker": base_fact_record.get("speaker", ""),
    }


def save_csv(rows: list[dict], path: Path):
    fieldnames = [
        "annotation_id",
        "example_id",
        "fact_id",
        "target_operation",
        "tags",
        "base_fact",
        "utterance",

        "is_valid",
        "error_type",
        "comment",

        "source_line_number",
        "prompt_version",
        "model",
        "source_dataset",
        "split",
        "dialogue_id",
        "session_id",
        "speaker",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            clean_item = dict(item)
            clean_item.pop("_source_line_number", None)
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")


def main():
    print(f"Loading: {INPUT_PATH}")

    records = load_jsonl(INPUT_PATH)

    print(f"Loaded records: {len(records)}")

    sampled_records = make_sample(records)

    annotation_rows = [
        flatten_for_annotation(item, annotation_id=i)
        for i, item in enumerate(sampled_records, start=1)
    ]

    save_csv(annotation_rows, OUTPUT_CSV_PATH)
    save_jsonl(sampled_records, OUTPUT_JSONL_PATH)

    print(f"Saved CSV for manual annotation: {OUTPUT_CSV_PATH}")
    print(f"Saved JSONL copy of sampled examples: {OUTPUT_JSONL_PATH}")
    print(f"Total sampled examples: {len(sampled_records)}")

    print("\nSample distribution:")

    counts = defaultdict(int)

    for item in sampled_records:
        counts[item.get("target_operation")] += 1

    for operation in SAMPLE_PLAN:
        print(f"- {operation}: {counts[operation]}")

    print("\nFill these columns manually in the CSV:")
    print("- is_valid: 1 or 0")
    print("- error_type: wrong_label / ambiguous / bad_generation / duplicate_like / other")
    print("- comment: optional")


if __name__ == "__main__":
    main()