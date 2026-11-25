import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("data/final/memory_operations_cleaned_2.jsonl")

OUTPUT_DIR = Path("data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSONL_PATH = OUTPUT_DIR / "memory_operations_cleaned_3_cross_filtered.jsonl"
REPORT_PATH = OUTPUT_DIR / "cross_operation_filtering_report.json"

RANDOM_SEED = 42

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

# Чистим только опасные пары классов.
# Именно между ними похожие предложения могут вредить обучению,
# потому что модель видит почти одинаковый текст с разными labels.
DANGEROUS_OPERATION_PAIRS = {
    frozenset(["DELETE", "CONFLICT"]),
    frozenset(["DELETE", "DELETE_ADD"]),
    frozenset(["CONFLICT", "DELETE_ADD"]),
}

# Порог похожести для cross-operation фильтрации.
# 0.85 — осторожный порог: удаляем только очень похожие пары.
# Если хочешь мягче — 0.90.
# Если хочешь строже — 0.80.
CROSS_OPERATION_JACCARD_THRESHOLD = 0.85

# После удаления снова балансируем классы по минимальному.
BALANCE_AFTER_FILTERING = True


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
    "now",
    "these days",
    "nowadays",
    "stopped",
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


DELETE_ADD_REPLACEMENT_MARKERS = [
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
    "now prefer",
    "now i prefer",
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
            cleaned.pop("_normalized_utterance", None)
            cleaned.pop("_content_lemmas", None)
            cleaned.pop("_content_lemma_set", None)
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
        operation = item.get("target_operation")

        if operation not in TARGET_OPERATIONS:
            continue

        if not utterance:
            continue

        lemmas = get_content_lemmas(utterance)

        item["_normalized_utterance"] = normalize_text(utterance)
        item["_content_lemmas"] = lemmas
        item["_content_lemma_set"] = set(lemmas)

        annotated.append(item)

    return annotated


def count_by_operation(records: list[dict]) -> dict[str, int]:
    counter = Counter()

    for item in records:
        counter[item.get("target_operation", "UNKNOWN")] += 1

    return dict(counter)


def group_by_operation(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    for item in records:
        grouped[item.get("target_operation")].append(item)

    return grouped


def choose_record_to_remove(item_a: dict, item_b: dict) -> tuple[dict, str]:
    """
    Решает, какой пример удалить из опасной cross-operation пары.

    Логика:
    1. Если CONFLICT содержит маркер изменения, удаляем CONFLICT.
    2. Если пара DELETE / DELETE_ADD и DELETE содержит replacement marker,
       удаляем DELETE, потому что он подозрительно похож на DELETE_ADD.
    3. Если один класс сейчас представлен больше другого, удаляем из более крупного.
       Это помогает меньше терять баланс.
    4. Иначе удаляем второй пример.
    """

    op_a = item_a.get("target_operation")
    op_b = item_b.get("target_operation")

    utt_a = get_utterance(item_a) or ""
    utt_b = get_utterance(item_b) or ""

    if op_a == "CONFLICT":
        markers = contains_phrase(utt_a, CONFLICT_CHANGE_MARKERS)
        if markers:
            return item_a, f"removed CONFLICT with change marker: {markers}"

    if op_b == "CONFLICT":
        markers = contains_phrase(utt_b, CONFLICT_CHANGE_MARKERS)
        if markers:
            return item_b, f"removed CONFLICT with change marker: {markers}"

    if {op_a, op_b} == {"DELETE", "DELETE_ADD"}:
        if op_a == "DELETE":
            markers = contains_phrase(utt_a, DELETE_ADD_REPLACEMENT_MARKERS)
            if markers:
                return item_a, f"removed DELETE with replacement marker: {markers}"

        if op_b == "DELETE":
            markers = contains_phrase(utt_b, DELETE_ADD_REPLACEMENT_MARKERS)
            if markers:
                return item_b, f"removed DELETE with replacement marker: {markers}"

    # Fallback: удаляем второй пример.
    return item_b, "removed second item in highly similar dangerous cross-operation pair"


def filter_cross_operation_near_duplicates(records: list[dict]):
    """
    Удаляет near-duplicates между опасными парами операций:
    DELETE / CONFLICT
    DELETE / DELETE_ADD
    CONFLICT / DELETE_ADD

    Важно:
    - ADD, IGNORE-NONFACT, IGNORE-OTHERS не трогаем в cross-operation режиме.
    - Внутри одного класса здесь ничего не удаляем.
    """

    grouped = group_by_operation(records)

    removed_ids = set()
    removed_records = []
    comparisons_checked = 0
    matched_pairs = []

    dangerous_pairs = [
        ("DELETE", "CONFLICT"),
        ("DELETE", "DELETE_ADD"),
        ("CONFLICT", "DELETE_ADD"),
    ]

    for op_a, op_b in dangerous_pairs:
        items_a = grouped.get(op_a, [])
        items_b = grouped.get(op_b, [])

        for item_a in items_a:
            if id(item_a) in removed_ids:
                continue

            lemmas_a = item_a.get("_content_lemma_set", set())

            for item_b in items_b:
                if id(item_b) in removed_ids:
                    continue

                comparisons_checked += 1

                lemmas_b = item_b.get("_content_lemma_set", set())

                score = jaccard(lemmas_a, lemmas_b)

                if score < CROSS_OPERATION_JACCARD_THRESHOLD:
                    continue

                pair_key = frozenset([op_a, op_b])

                if pair_key not in DANGEROUS_OPERATION_PAIRS:
                    continue

                to_remove, reason = choose_record_to_remove(item_a, item_b)

                removed_ids.add(id(to_remove))
                removed_records.append(to_remove)

                kept = item_b if to_remove is item_a else item_a

                matched_pairs.append({
                    "operation_pair": f"{op_a} / {op_b}",
                    "jaccard": round(score, 4),
                    "shared_lemmas": sorted(lemmas_a & lemmas_b),

                    "removed_reason": reason,

                    "removed_example_id": to_remove.get("example_id", ""),
                    "removed_fact_id": to_remove.get("fact_id", ""),
                    "removed_target_operation": to_remove.get("target_operation", ""),
                    "removed_utterance": get_utterance(to_remove),
                    "removed_source_line_number": to_remove.get("_source_line_number", None),

                    "kept_example_id": kept.get("example_id", ""),
                    "kept_fact_id": kept.get("fact_id", ""),
                    "kept_target_operation": kept.get("target_operation", ""),
                    "kept_utterance": get_utterance(kept),
                    "kept_source_line_number": kept.get("_source_line_number", None),
                })

                # Если item_a удалили, нет смысла сравнивать его дальше.
                if to_remove is item_a:
                    break

    filtered = [
        item for item in records
        if id(item) not in removed_ids
    ]

    return filtered, removed_records, matched_pairs, comparisons_checked


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
        balanced.extend(items[:min_count])

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

    balancing_info = {
        "examples_per_class_before_final_balancing": counts_before,
        "balanced_examples_per_class": min_count,
        "examples_per_class_final": {
            operation: min_count
            for operation in TARGET_OPERATIONS
        },
    }

    return balanced, balancing_info


def main():
    print(f"Loading: {INPUT_PATH}")

    records = load_jsonl(INPUT_PATH)
    print(f"Loaded records: {len(records)}")

    records = annotate_records(records)
    print(f"Valid annotated records: {len(records)}")

    original_counts = count_by_operation(records)

    filtered_records, removed_records, matched_pairs, comparisons_checked = (
        filter_cross_operation_near_duplicates(records)
    )

    print(f"Comparisons checked: {comparisons_checked}")
    print(f"Removed cross-operation near-duplicates: {len(removed_records)}")
    print(f"After cross-operation filtering: {len(filtered_records)}")

    if BALANCE_AFTER_FILTERING:
        final_records, balancing_info = balance_by_min_class(filtered_records)
    else:
        final_records = filtered_records
        balancing_info = {
            "examples_per_class_before_final_balancing": count_by_operation(filtered_records),
            "balanced_examples_per_class": None,
            "examples_per_class_final": count_by_operation(filtered_records),
        }

    print(f"Final records: {len(final_records)}")

    write_jsonl(final_records, OUTPUT_JSONL_PATH)

    report = {
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_JSONL_PATH),
        "random_seed": RANDOM_SEED,

        "settings": {
            "cross_operation_jaccard_threshold": CROSS_OPERATION_JACCARD_THRESHOLD,
            "dangerous_operation_pairs": [
                sorted(list(pair))
                for pair in DANGEROUS_OPERATION_PAIRS
            ],
            "balance_after_filtering": BALANCE_AFTER_FILTERING,
            "note": (
                "This script filters only highly similar examples across dangerous "
                "operation pairs, mainly DELETE/CONFLICT, DELETE/DELETE_ADD, "
                "and CONFLICT/DELETE_ADD. Previous dataset versions are not modified."
            ),
        },

        "original_examples": len(records),
        "examples_per_class_original": original_counts,

        "comparisons_checked": comparisons_checked,
        "cross_operation_near_duplicates_removed": len(removed_records),
        "after_cross_operation_filtering": len(filtered_records),
        "examples_per_class_after_cross_operation_filtering": count_by_operation(filtered_records),

        **balancing_info,

        "final_examples": len(final_records),
        "final_distribution": count_by_operation(final_records),

        "removed_cross_operation_pairs_preview_first_300": matched_pairs[:300],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset: {OUTPUT_JSONL_PATH}")
    print(f"Saved report: {REPORT_PATH}")

    print("\nFinal class distribution:")
    final_counts = count_by_operation(final_records)

    for operation in TARGET_OPERATIONS:
        print(f"- {operation}: {final_counts.get(operation, 0)}")

    print("\nDone.")


if __name__ == "__main__":
    main()