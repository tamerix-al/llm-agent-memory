import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("data/generated/single_operation_test_v2.jsonl")

OUTPUT_DIR = Path("data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL_PATH = OUTPUT_DIR / "memory_operations_clean_balanced.jsonl"
CLEANING_REPORT_PATH = OUTPUT_DIR / "cleaning_report.json"

RANDOM_SEED = 42

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]


def normalize_utterance(text: str) -> str:
    """
    Нормализация для поиска точных дублей.

    Например:
    "I hate running."
    " i hate running "
    будут считаться одинаковыми.

    Но:
    "I hate running."
    "I absolutely hate running."
    будут считаться разными.
    """
    text = str(text).strip().lower()
    text = " ".join(text.split())

    # Убираем финальную пунктуацию, чтобы:
    # "Oh, that's nice!"
    # "Oh, that's nice."
    # считались одинаковыми.
    text = text.rstrip(".!?")

    return text


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                item["_source_line_number"] = line_number
                records.append(item)

            except json.JSONDecodeError as e:
                print(f"Skipping broken JSON line {line_number}: {e}")

    return records


def get_utterance(item: dict) -> str | None:
    """
    Достаём utterance.

    В новом single-operation формате utterance лежит на верхнем уровне.
    Но на всякий случай поддерживаем parsed_response.
    """
    utterance = item.get("utterance")

    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()

    parsed = item.get("parsed_response")

    if isinstance(parsed, dict):
        utterance = parsed.get("utterance")

        if isinstance(utterance, str) and utterance.strip():
            return utterance.strip()

    return None


def is_valid_item(item: dict) -> bool:
    """
    Базовая проверка, что пример можно оставить в датасете.
    """
    target_operation = item.get("target_operation")
    utterance = get_utterance(item)

    if target_operation not in TARGET_OPERATIONS:
        return False

    if not utterance:
        return False

    return True


def deduplicate_within_operations(records: list[dict]):
    """
    Удаляет точные дубли внутри каждого target_operation.

    То есть:
    [CONFLICT] I hate running.
    [CONFLICT] I hate running.
    -> останется один

    Но:
    [CONFLICT] I hate running.
    [DELETE] I hate running.
    -> оба останутся, потому что это разные классы.
       Такие случаи лучше потом смотреть отдельно как возможные ошибки разметки.
    """
    seen = set()
    deduplicated = []
    duplicate_records = []

    for item in records:
        target_operation = item.get("target_operation")
        utterance = get_utterance(item)
        normalized = normalize_utterance(utterance)

        key = (target_operation, normalized)

        if key in seen:
            duplicate_records.append(item)
            continue

        seen.add(key)
        deduplicated.append(item)

    return deduplicated, duplicate_records


def group_by_operation(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    for item in records:
        grouped[item["target_operation"]].append(item)

    return grouped


def balance_by_min_class(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Балансирует датасет по минимальному классу.

    Например, после удаления дублей осталось:
    ADD: 492
    DELETE: 460
    CONFLICT: 455

    Тогда берём по 455 из каждого класса.
    """
    grouped = group_by_operation(records)

    counts_before = {
        operation: len(grouped.get(operation, []))
        for operation in TARGET_OPERATIONS
    }

    min_count = min(counts_before.values())

    random.seed(RANDOM_SEED)

    balanced = []

    selected_ids_by_operation = {}

    for operation in TARGET_OPERATIONS:
        items = grouped.get(operation, [])

        # Перемешиваем, чтобы не брать всегда первые примеры.
        items = items.copy()
        random.shuffle(items)

        selected = items[:min_count]

        selected_ids_by_operation[operation] = [
            item.get("example_id", "")
            for item in selected
        ]

        balanced.extend(selected)

    # Для удобства сортируем финальный файл:
    # сначала по fact_id, потом по порядку операций.
    operation_order = {
        operation: index
        for index, operation in enumerate(TARGET_OPERATIONS)
    }

    balanced.sort(
        key=lambda item: (
            str(item.get("fact_id", "")),
            operation_order.get(item.get("target_operation", ""), 999),
            str(item.get("example_id", ""))
        )
    )

    balancing_info = {
        "examples_per_class_after_deduplication_before_balancing": counts_before,
        "balanced_examples_per_class": min_count,
        "selected_examples_by_operation": {
            operation: len(ids)
            for operation, ids in selected_ids_by_operation.items()
        }
    }

    return balanced, balancing_info


def remove_internal_fields(item: dict) -> dict:
    """
    Убираем служебные поля, которые нужны только внутри cleaning-скрипта.
    """
    cleaned = dict(item)
    cleaned.pop("_source_line_number", None)
    return cleaned


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            item = remove_internal_fields(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_by_operation(records: list[dict]) -> dict:
    counter = Counter()

    for item in records:
        counter[item.get("target_operation", "UNKNOWN")] += 1

    return dict(counter)


def main():
    print(f"Loading: {INPUT_PATH}")

    records = load_jsonl(INPUT_PATH)

    original_count = len(records)

    print(f"Original records: {original_count}")

    valid_records = []
    invalid_records = []

    for item in records:
        if is_valid_item(item):
            valid_records.append(item)
        else:
            invalid_records.append(item)

    print(f"Valid records: {len(valid_records)}")
    print(f"Invalid records: {len(invalid_records)}")

    before_dedup_counts = count_by_operation(valid_records)

    deduplicated_records, duplicate_records = deduplicate_within_operations(valid_records)

    print(f"After exact deduplication: {len(deduplicated_records)}")
    print(f"Removed exact duplicates: {len(duplicate_records)}")

    after_dedup_counts = count_by_operation(deduplicated_records)

    balanced_records, balancing_info = balance_by_min_class(deduplicated_records)

    print(f"Final balanced records: {len(balanced_records)}")
    print(f"Output JSONL: {OUTPUT_JSONL_PATH}")

    write_jsonl(balanced_records, OUTPUT_JSONL_PATH)

    duplicate_examples_preview = []

    for item in duplicate_records[:100]:
        duplicate_examples_preview.append({
            "example_id": item.get("example_id", ""),
            "fact_id": item.get("fact_id", ""),
            "target_operation": item.get("target_operation", ""),
            "utterance": get_utterance(item),
            "source_line_number": item.get("_source_line_number", None)
        })

    invalid_examples_preview = []

    for item in invalid_records[:100]:
        invalid_examples_preview.append({
            "example_id": item.get("example_id", ""),
            "fact_id": item.get("fact_id", ""),
            "target_operation": item.get("target_operation", ""),
            "utterance": get_utterance(item),
            "parse_status": item.get("parse_status", ""),
            "source_line_number": item.get("_source_line_number", None)
        })

    report = {
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_JSONL_PATH),
        "random_seed": RANDOM_SEED,

        "original_examples": original_count,
        "valid_examples": len(valid_records),
        "invalid_examples": len(invalid_records),

        "examples_per_class_before_deduplication": before_dedup_counts,

        "after_exact_deduplication": len(deduplicated_records),
        "removed_exact_duplicates": len(duplicate_records),
        "examples_per_class_after_deduplication": after_dedup_counts,

        **balancing_info,

        "final_examples": len(balanced_records),
        "examples_per_class_final": count_by_operation(balanced_records),

        "duplicate_examples_preview_first_100": duplicate_examples_preview,
        "invalid_examples_preview_first_100": invalid_examples_preview,
    }

    with open(CLEANING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Cleaning report: {CLEANING_REPORT_PATH}")

    print("\nFinal class distribution:")
    for operation, count in count_by_operation(balanced_records).items():
        print(f"- {operation}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()