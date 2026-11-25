import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEST_CSV_PATH = Path("data/baseline_classifier/test_predictions.csv")
BASE_FACTS_PATH = Path("data/processed/base_facts_upd.jsonl")

OUTPUT_DIR = Path("data/multifact_context")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FACTS_OUTPUT_PATH = OUTPUT_DIR / "multifact_all_user_facts.jsonl"
TOPK_OUTPUT_PATH = OUTPUT_DIR / "multifact_retrieved_topk_facts.jsonl"

RANDOM_SEED = 42
TOP_K = 5

# Для DELETE / DELETE_ADD / CONFLICT target fact должен быть в контексте,
# иначе модель физически не сможет выбрать правильный target_fact_id.
# Поэтому в retrieved_top_k варианте мы форсируем включение target fact,
# если retrieval сам его не выбрал.
FORCE_INCLUDE_TARGET_IN_TOPK = True

OPERATIONS_WITH_TARGET = {
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
}

OPERATIONS_WITHOUT_TARGET = {
    "ADD",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
}

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_dialogue_id(item: dict[str, Any]) -> str:
    """
    В MSC поле иногда могло называться dialogue_id или dialoug_id.
    Поддерживаем оба варианта.
    """
    if item.get("dialogue_id") is not None:
        return str(item.get("dialogue_id"))

    if item.get("dialoug_id") is not None:
        return str(item.get("dialoug_id"))

    base_fact_record = item.get("base_fact_record")

    if isinstance(base_fact_record, dict):
        if base_fact_record.get("dialogue_id") is not None:
            return str(base_fact_record.get("dialogue_id"))

        if base_fact_record.get("dialoug_id") is not None:
            return str(base_fact_record.get("dialoug_id"))

    return "UNKNOWN_DIALOGUE"


def get_speaker(item: dict[str, Any]) -> str:
    if item.get("speaker") is not None:
        return str(item.get("speaker"))

    base_fact_record = item.get("base_fact_record")

    if isinstance(base_fact_record, dict):
        if base_fact_record.get("speaker") is not None:
            return str(base_fact_record.get("speaker"))

    return "UNKNOWN_SPEAKER"


def get_user_key(item: dict[str, Any]) -> tuple[str, str]:
    """
    Один пользователь в нашем setup определяется как:
    dialogue_id + speaker.
    """
    return get_dialogue_id(item), get_speaker(item)


def build_user_fact_index(
    base_fact_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """
    Возвращает:
    1. fact_id -> fact record
    2. (dialogue_id, speaker) -> список фактов этого пользователя
    """
    fact_id_to_record = {}
    user_to_facts = {}

    for item in base_fact_records:
        fact_id = str(item.get("fact_id", "")).strip()
        base_fact = str(item.get("base_fact", "")).strip()

        if not fact_id or not base_fact:
            continue

        user_key = get_user_key(item)

        clean_record = {
            "fact_id": fact_id,
            "fact": base_fact,
            "dialogue_id": user_key[0],
            "speaker": user_key[1],
            "source_record": item,
        }

        fact_id_to_record[fact_id] = clean_record

        if user_key not in user_to_facts:
            user_to_facts[user_key] = []

        user_to_facts[user_key].append(clean_record)

    # Дедупликация фактов внутри пользователя
    for user_key, facts in user_to_facts.items():
        seen = set()
        unique_facts = []

        for fact in facts:
            key = (fact["fact_id"], fact["fact"].lower())

            if key in seen:
                continue

            seen.add(key)
            unique_facts.append(fact)

        user_to_facts[user_key] = unique_facts

    return fact_id_to_record, user_to_facts


def deterministic_shuffle(items: list[dict[str, Any]], seed_value: str) -> list[dict[str, Any]]:
    items = list(items)
    local_random = random.Random(seed_value)
    local_random.shuffle(items)
    return items


def assign_prompt_fact_ids(
    context_facts: list[dict[str, Any]],
    gold_original_fact_id: str | None,
    operation: str,
) -> tuple[list[dict[str, Any]], int | None, str | None]:
    """
    Превращает original fact ids в prompt-local ids: [1], [2], [3]...
    """
    memory_facts = []
    gold_prompt_fact_id = None
    gold_target_fact = None

    for idx, fact in enumerate(context_facts, start=1):
        original_fact_id = fact["fact_id"]
        fact_text = fact["fact"]

        is_target = (
            operation in OPERATIONS_WITH_TARGET
            and gold_original_fact_id is not None
            and original_fact_id == gold_original_fact_id
        )

        memory_facts.append({
            "prompt_fact_id": idx,
            "original_fact_id": original_fact_id,
            "fact": fact_text,
            "is_gold_target": is_target,
        })

        if is_target:
            gold_prompt_fact_id = idx
            gold_target_fact = fact_text

    return memory_facts, gold_prompt_fact_id, gold_target_fact


def make_all_user_context(
    row: dict[str, Any],
    fact_id_to_record: dict[str, dict[str, Any]],
    user_to_facts: dict[tuple[str, str], list[dict[str, Any]]],
    row_index: int,
) -> dict[str, Any] | None:
    fact_id = str(row.get("fact_id", "")).strip()
    operation = str(row.get("target_operation", "")).strip()

    if fact_id not in fact_id_to_record:
        return None

    target_fact_record = fact_id_to_record[fact_id]
    user_key = (
        target_fact_record["dialogue_id"],
        target_fact_record["speaker"],
    )

    context_facts = user_to_facts.get(user_key, [])

    if not context_facts:
        return None

    context_facts = deterministic_shuffle(
        context_facts,
        seed_value=f"all::{RANDOM_SEED}::{row_index}::{fact_id}",
    )

    gold_original_fact_id = fact_id if operation in OPERATIONS_WITH_TARGET else None

    memory_facts, gold_prompt_fact_id, gold_target_fact = assign_prompt_fact_ids(
        context_facts=context_facts,
        gold_original_fact_id=gold_original_fact_id,
        operation=operation,
    )

    return {
        "example_id": str(row.get("example_id", "")),
        "source_fact_id": fact_id,
        "base_fact": str(row.get("base_fact", "")),
        "utterance": str(row.get("utterance", "")),
        "gold_operation": operation,
        "gold_target_fact_id": gold_prompt_fact_id,
        "gold_target_fact": gold_target_fact,
        "context_type": "all_user_facts",
        "num_memory_facts": len(memory_facts),
        "memory_facts": memory_facts,
        "source_test_row": row,
        "user_key": {
            "dialogue_id": user_key[0],
            "speaker": user_key[1],
        },
    }


def retrieve_top_k_facts(
    utterance: str,
    facts: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Простая retrieval-стратегия:
    TF-IDF cosine similarity между utterance и каждым фактом пользователя.
    """
    if not facts:
        return [], {}

    fact_texts = [fact["fact"] for fact in facts]
    texts = [utterance] + fact_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(texts)

    utterance_vector = matrix[0]
    fact_vectors = matrix[1:]

    similarities = cosine_similarity(utterance_vector, fact_vectors)[0]

    scored_facts = []

    similarity_by_fact_id = {}

    for fact, score in zip(facts, similarities):
        score = float(score)
        similarity_by_fact_id[fact["fact_id"]] = score
        scored_facts.append((fact, score))

    scored_facts.sort(key=lambda x: x[1], reverse=True)

    selected = [fact for fact, _ in scored_facts[:top_k]]

    return selected, similarity_by_fact_id


def make_topk_context(
    row: dict[str, Any],
    fact_id_to_record: dict[str, dict[str, Any]],
    user_to_facts: dict[tuple[str, str], list[dict[str, Any]]],
    row_index: int,
) -> dict[str, Any] | None:
    fact_id = str(row.get("fact_id", "")).strip()
    operation = str(row.get("target_operation", "")).strip()
    utterance = str(row.get("utterance", "")).strip()

    if fact_id not in fact_id_to_record:
        return None

    target_fact_record = fact_id_to_record[fact_id]
    user_key = (
        target_fact_record["dialogue_id"],
        target_fact_record["speaker"],
    )

    all_user_facts = user_to_facts.get(user_key, [])

    if not all_user_facts:
        return None

    selected_facts, similarity_by_fact_id = retrieve_top_k_facts(
        utterance=utterance,
        facts=all_user_facts,
        top_k=TOP_K,
    )

    target_required = operation in OPERATIONS_WITH_TARGET
    target_was_in_raw_topk = fact_id in {fact["fact_id"] for fact in selected_facts}

    if (
        FORCE_INCLUDE_TARGET_IN_TOPK
        and target_required
        and not target_was_in_raw_topk
    ):
        # Заменяем последний факт на target fact, чтобы задача была разрешимой.
        selected_fact_ids = {fact["fact_id"] for fact in selected_facts}

        if fact_id not in selected_fact_ids:
            if len(selected_facts) >= TOP_K:
                selected_facts = selected_facts[:-1]

            selected_facts.append(target_fact_record)

    selected_facts = deterministic_shuffle(
        selected_facts,
        seed_value=f"topk::{RANDOM_SEED}::{row_index}::{fact_id}",
    )

    gold_original_fact_id = fact_id if target_required else None

    memory_facts, gold_prompt_fact_id, gold_target_fact = assign_prompt_fact_ids(
        context_facts=selected_facts,
        gold_original_fact_id=gold_original_fact_id,
        operation=operation,
    )

    target_similarity = similarity_by_fact_id.get(fact_id)

    return {
        "example_id": str(row.get("example_id", "")),
        "source_fact_id": fact_id,
        "base_fact": str(row.get("base_fact", "")),
        "utterance": utterance,
        "gold_operation": operation,
        "gold_target_fact_id": gold_prompt_fact_id,
        "gold_target_fact": gold_target_fact,
        "context_type": "retrieved_topk_facts",
        "retrieval_method": "tfidf_cosine",
        "top_k": TOP_K,
        "force_include_target": FORCE_INCLUDE_TARGET_IN_TOPK,
        "target_was_in_raw_topk": target_was_in_raw_topk if target_required else None,
        "target_similarity": target_similarity,
        "num_memory_facts": len(memory_facts),
        "memory_facts": memory_facts,
        "source_test_row": row,
        "user_key": {
            "dialogue_id": user_key[0],
            "speaker": user_key[1],
        },
    }


def print_example(record: dict[str, Any]):
    print("\nEXAMPLE")
    print("=======")
    print(f"Context type: {record['context_type']}")
    print(f"Gold operation: {record['gold_operation']}")
    print(f"Gold target fact id: {record['gold_target_fact_id']}")
    print()

    print("Memory facts:")
    for fact in record["memory_facts"]:
        marker = "  <-- gold target" if fact["is_gold_target"] else ""
        print(f"[{fact['prompt_fact_id']}] {fact['fact']}{marker}")

    print("\nUtterance:")
    print(record["utterance"])

    print("\nExpected output:")
    print(json.dumps({
        "operation": record["gold_operation"],
        "target_fact_id": record["gold_target_fact_id"],
    }, ensure_ascii=False, indent=2))


def main():
    print(f"Loading test split: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)

    print(f"Loading base facts: {BASE_FACTS_PATH}")
    base_fact_records = load_jsonl(BASE_FACTS_PATH)

    fact_id_to_record, user_to_facts = build_user_fact_index(base_fact_records)

    print(f"Base facts loaded: {len(fact_id_to_record)}")
    print(f"Users found: {len(user_to_facts)}")
    print(f"Test examples loaded: {len(test_df)}")

    all_context_records = []
    topk_context_records = []

    skipped_all = 0
    skipped_topk = 0

    for row_index, row in test_df.iterrows():
        row_dict = row.to_dict()

        operation = str(row_dict.get("target_operation", "")).strip()

        if operation not in TARGET_OPERATIONS:
            continue

        all_record = make_all_user_context(
            row=row_dict,
            fact_id_to_record=fact_id_to_record,
            user_to_facts=user_to_facts,
            row_index=row_index,
        )

        if all_record is None:
            skipped_all += 1
        else:
            all_context_records.append(all_record)

        topk_record = make_topk_context(
            row=row_dict,
            fact_id_to_record=fact_id_to_record,
            user_to_facts=user_to_facts,
            row_index=row_index,
        )

        if topk_record is None:
            skipped_topk += 1
        else:
            topk_context_records.append(topk_record)

    save_jsonl(all_context_records, ALL_FACTS_OUTPUT_PATH)
    save_jsonl(topk_context_records, TOPK_OUTPUT_PATH)

    print("\nSaved:")
    print(f"- {ALL_FACTS_OUTPUT_PATH}: {len(all_context_records)} records")
    print(f"- {TOPK_OUTPUT_PATH}: {len(topk_context_records)} records")

    print("\nSkipped:")
    print(f"- all_user_facts: {skipped_all}")
    print(f"- retrieved_topk_facts: {skipped_topk}")

    if all_context_records:
        print_example(all_context_records[0])

    if topk_context_records:
        print_example(topk_context_records[0])

    update_records = [
        item for item in topk_context_records
        if item["gold_operation"] in OPERATIONS_WITH_TARGET
    ]

    if update_records:
        raw_topk_hits = [
            item["target_was_in_raw_topk"]
            for item in update_records
            if item["target_was_in_raw_topk"] is not None
        ]

        if raw_topk_hits:
            retrieval_recall = sum(raw_topk_hits) / len(raw_topk_hits)
            print(f"\nRaw top-{TOP_K} retrieval recall for target facts: {retrieval_recall:.4f}")


if __name__ == "__main__":
    main()