import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


INPUT_PATH = Path("data/final/memory_operations_clean_balanced.jsonl")

OUTPUT_DIR = Path("data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL_PATH = OUTPUT_DIR / "memory_operations_cleaned_2.jsonl"
FILTERING_REPORT_PATH = OUTPUT_DIR / "filtering_report.json"

RANDOM_SEED = 42

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

# Порог для удаления почти-дублей.
# 0.85 — довольно мягкий порог: удаляются только очень похожие пары.
# Если хочешь строже, можно поставить 0.80.
# Если хочешь осторожнее, можно поставить 0.90.
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85

# Нужно ли автоматически удалять CONFLICT-примеры, где есть change/update markers.
# Я бы оставил True, потому что CONFLICT должен быть противоречием без сигнала изменения.
FILTER_SUSPICIOUS_CONFLICT = True

# DELETE-примеры НЕ удаляем по suspicious rules.
# Мы только флагируем их в отчёте, потому что ты прав:
# "anymore", "now", "no longer" могут быть хорошими DELETE-маркерами.
FILTER_SUSPICIOUS_DELETE = False


STOPWORDS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "a", "an", "the",
    "and", "or", "but", "if", "because", "so",
    "to", "of", "in", "on", "at", "for", "from", "with", "about",
    "as", "by", "into", "over", "after", "before", "during",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "may", "might", "must",
    "not", "no",
    "this", "that", "these", "those",
    "there", "here",
    "just", "really", "very", "quite", "also", "actually",
    "usually", "often", "sometimes",
}


CONFLICT_CHANGE_MARKERS = [
    "anymore",
    "no longer",
    "used to",
    "i used to",
    "now",
    "these days",
    "nowadays",
    "stopped",
    "stop ",
    "quit",
    "left",
    "moved",
    "switched",
    "started",
    "began",
    "became",
    "decided against",
    "gave up",
    "given up",
]

# Для DELETE это НЕ авто-удаление.
# Это только флаги в report, чтобы можно было руками посмотреть случаи,
# где DELETE может быть ближе к DELETE_ADD.
DELETE_REPLACEMENT_CUES = [
    "instead",
    "switched to",
    "started",
    "began",
    "became",
    "moved to",
    "adopted",
    "bought",
    "purchased",
    "got a",
    "got an",
    "joined",
    "signed up",
    "prefer",
    "now i prefer",
    "now prefer",
]


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


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            cleaned = dict(item)
            cleaned.pop("_source_line_number", None)
            cleaned.pop("_content_lemmas", None)
            cleaned.pop("_normalized_utterance", None)
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def get_utterance(item: dict) -> str | None:
    utterance = item.get("utterance")

    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()

    parsed = item.get("parsed_response")

    if isinstance(parsed, dict):
        utterance = parsed.get("utterance")

        if isinstance(utterance, str) and utterance.strip():
            return utterance.strip()

    return None


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = " ".join(text.split())
    text = text.rstrip(".!?")
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def simple_lemma(token: str) -> str:
    token = token.lower().strip("'")

    irregular = {
        "went": "go",
        "gone": "go",
        "better": "good",
        "best": "good",
        "worse": "bad",
        "worst": "bad",
        "children": "child",
        "people": "person",
        "men": "man",
        "women": "woman",
        "mice": "mouse",
        "feet": "foot",
        "teeth": "tooth",
    }

    if token in irregular:
        return irregular[token]

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("es"):
        return token[:-2]

    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token


def get_content_lemmas(text: str) -> list[str]:
    tokens = tokenize(text)
    lemmas = []

    for token in tokens:
        lemma = simple_lemma(token)

        if not lemma:
            continue

        if lemma in STOPWORDS:
            continue

        if len(lemma) <= 2:
            continue

        lemmas.append(lemma)

    return lemmas


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0

    union = set_a | set_b

    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


def count_by_operation(records: list[dict]) -> dict[str, int]:
    counter = Counter()

    for item in records:
        counter[item.get("target_operation", "UNKNOWN")] += 1

    return dict(counter)


def group_by_operation(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    for item in records:
        operation = item.get("target_operation")
        grouped[operation].append(item)

    return grouped


def contains_phrase(text: str, phrases: list[str]) -> list[str]:
    text_norm = normalize_text(text)
    found = []

    for phrase in phrases:
        phrase_norm = normalize_text(phrase)

        if phrase_norm in text_norm:
            found.append(phrase)

    return found


def annotate_records(records: list[dict]) -> list[dict]:
    annotated = []

    for item in records:
        utterance = get_utterance(item)

        if not utterance:
            continue

        target_operation = item.get("target_operation")

        if target_operation not in TARGET_OPERATIONS:
            continue

        item["_normalized_utterance"] = normalize_text(utterance)
        item["_content_lemmas"] = get_content_lemmas(utterance)

        annotated.append(item)

    return annotated


def find_suspicious_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Возвращает:
    - records_to_remove: то, что реально удаляем
    - suspicious_preview: то, что просто записываем в report

    Важное решение:
    DELETE автоматически НЕ удаляется.
    DELETE с "anymore", "now", "no longer" может быть нормальным.
    """
    records_to_remove = []
    suspicious_preview = []

    for item in records:
        operation = item.get("target_operation")
        utterance = get_utterance(item) or ""

        if operation == "CONFLICT":
            found_markers = contains_phrase(utterance, CONFLICT_CHANGE_MARKERS)

            if found_markers:
                suspicious_preview.append({
                    "action": "removed" if FILTER_SUSPICIOUS_CONFLICT else "flagged_only",
                    "reason": "CONFLICT contains change/update marker",
                    "matched_markers": found_markers,
                    "example_id": item.get("example_id", ""),
                    "fact_id": item.get("fact_id", ""),
                    "target_operation": operation,
                    "utterance": utterance,
                    "source_line_number": item.get("_source_line_number", None),
                })

                if FILTER_SUSPICIOUS_CONFLICT:
                    records_to_remove.append(item)

        elif operation == "DELETE":
            found_cues = contains_phrase(utterance, DELETE_REPLACEMENT_CUES)

            if found_cues:
                suspicious_preview.append({
                    "action": "flagged_only" if not FILTER_SUSPICIOUS_DELETE else "removed",
                    "reason": "DELETE may introduce replacement fact, review manually",
                    "matched_markers": found_cues,
                    "example_id": item.get("example_id", ""),
                    "fact_id": item.get("fact_id", ""),
                    "target_operation": operation,
                    "utterance": utterance,
                    "source_line_number": item.get("_source_line_number", None),
                })

                if FILTER_SUSPICIOUS_DELETE:
                    records_to_remove.append(item)

    return records_to_remove, suspicious_preview


def remove_near_duplicates_within_operation(records: list[dict]):
    """
    Удаляет почти-дубли внутри каждого target_operation.

    Логика:
    - группируем по операции
    - идём по примерам в случайном, но воспроизводимом порядке
    - оставляем пример, если он не слишком похож на уже оставленные
    - если похожесть >= threshold, отправляем в removed
    """
    random.seed(RANDOM_SEED)

    grouped = group_by_operation(records)

    kept_all = []
    removed_near_duplicates = []

    for operation in TARGET_OPERATIONS:
        items = grouped.get(operation, []).copy()

        # Перемешиваем, чтобы не всегда выигрывали первые строки.
        random.shuffle(items)

        kept_for_operation = []

        for item in items:
            item_lemmas = set(item.get("_content_lemmas", []))

            duplicate_of = None
            duplicate_score = 0.0
            duplicate_shared_lemmas = []

            for kept_item in kept_for_operation:
                kept_lemmas = set(kept_item.get("_content_lemmas", []))

                score = jaccard(item_lemmas, kept_lemmas)

                if score >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                    duplicate_of = kept_item
                    duplicate_score = score
                    duplicate_shared_lemmas = sorted(item_lemmas & kept_lemmas)
                    break

            if duplicate_of is None:
                kept_for_operation.append(item)
            else:
                removed_near_duplicates.append({
                    "removed_example_id": item.get("example_id", ""),
                    "removed_fact_id": item.get("fact_id", ""),
                    "target_operation": operation,
                    "removed_utterance": get_utterance(item),

                    "kept_example_id": duplicate_of.get("example_id", ""),
                    "kept_fact_id": duplicate_of.get("fact_id", ""),
                    "kept_utterance": get_utterance(duplicate_of),

                    "jaccard": round(duplicate_score, 4),
                    "shared_lemmas": duplicate_shared_lemmas,
                    "removed_source_line_number": item.get("_source_line_number", None),
                    "kept_source_line_number": duplicate_of.get("_source_line_number", None),
                })

        kept_all.extend(kept_for_operation)

    return kept_all, removed_near_duplicates


def balance_by_min_class(records: list[dict]) -> tuple[list[dict], dict]:
    grouped = group_by_operation(records)

    counts_before = {
        operation: len(grouped.get(operation, []))
        for operation in TARGET_OPERATIONS
    }

    min_count = min(counts_before.values())

    random.seed(RANDOM_SEED)

    balanced = []

    for operation in TARGET_OPERATIONS:
        items = grouped.get(operation, []).copy()
        random.shuffle(items)

        selected = items[:min_count]
        balanced.extend(selected)

    operation_order = {
        operation: index
        for index, operation in enumerate(TARGET_OPERATIONS)
    }

    balanced.sort(
        key=lambda item: (
            str(item.get("fact_id", "")),
            operation_order.get(item.get("target_operation", ""), 999),
            str(item.get("example_id", "")),
        )
    )

    info = {
        "examples_per_class_before_balancing": counts_before,
        "balanced_examples_per_class": min_count,
        "examples_per_class_final": {
            operation: min_count
            for operation in TARGET_OPERATIONS
        },
    }

    return balanced, info


def main():
    print(f"Loading: {INPUT_PATH}")

    records = load_jsonl(INPUT_PATH)
    print(f"Loaded records: {len(records)}")

    records = annotate_records(records)
    print(f"Valid annotated records: {len(records)}")

    original_counts = count_by_operation(records)

    suspicious_to_remove, suspicious_preview = find_suspicious_records(records)

    ids_to_remove = {id(item) for item in suspicious_to_remove}

    after_suspicious_filter = [
        item for item in records
        if id(item) not in ids_to_remove
    ]

    print(f"After suspicious filtering: {len(after_suspicious_filter)}")
    print(f"Suspicious records removed: {len(suspicious_to_remove)}")
    print(f"Suspicious records flagged in report: {len(suspicious_preview)}")

    after_near_dedup, removed_near_duplicates = remove_near_duplicates_within_operation(
        after_suspicious_filter
    )

    print(f"After near-duplicate filtering: {len(after_near_dedup)}")
    print(f"Near-duplicates removed: {len(removed_near_duplicates)}")

    balanced_records, balancing_info = balance_by_min_class(after_near_dedup)

    print(f"Final balanced records: {len(balanced_records)}")

    write_jsonl(balanced_records, OUTPUT_JSONL_PATH)

    report = {
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_JSONL_PATH),
        "random_seed": RANDOM_SEED,

        "settings": {
            "near_duplicate_jaccard_threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
            "filter_suspicious_conflict": FILTER_SUSPICIOUS_CONFLICT,
            "filter_suspicious_delete": FILTER_SUSPICIOUS_DELETE,
            "note": (
                "DELETE examples are not automatically removed for markers like "
                "'anymore', 'now', or 'no longer', because these can be valid deletion markers."
            ),
        },

        "original_examples": len(records),
        "examples_per_class_original": original_counts,

        "after_suspicious_filtering": len(after_suspicious_filter),
        "suspicious_records_removed": len(suspicious_to_remove),
        "suspicious_records_flagged_total": len(suspicious_preview),

        "after_near_duplicate_filtering": len(after_near_dedup),
        "near_duplicates_removed": len(removed_near_duplicates),

        **balancing_info,

        "final_examples": len(balanced_records),
        "final_distribution": count_by_operation(balanced_records),

        "suspicious_records_preview_first_200": suspicious_preview[:200],
        "removed_near_duplicates_preview_first_200": removed_near_duplicates[:200],
    }

    with open(FILTERING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved filtered balanced dataset: {OUTPUT_JSONL_PATH}")
    print(f"Saved filtering report: {FILTERING_REPORT_PATH}")

    print("\nFinal class distribution:")
    final_counts = count_by_operation(balanced_records)

    for operation in TARGET_OPERATIONS:
        print(f"- {operation}: {final_counts.get(operation, 0)}")

    print("\nDone.")


if __name__ == "__main__":
    main()