import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


TEST_CSV_PATH = Path("data/baseline_classifier/test_predictions.csv")
BASE_FACTS_PATH = Path("data/processed/base_facts_upd.jsonl")

OUTPUT_DIR = Path("data/multifact_context")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "multifact_mpnet_topk_facts.jsonl"
REPORT_PATH = OUTPUT_DIR / "mpnet_topk_report.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

RANDOM_SEED = 42
TOP_K = 5
BATCH_SIZE = 16

# Для DELETE / DELETE_ADD / CONFLICT нужен релевантный исходный факт.
# Поэтому, если retrieval сам не нашёл target fact, мы добавляем его принудительно.
# Это делает эксперимент контролируемым: мы проверяем влияние шума, а не качество retrieval как отдельной задачи.
FORCE_INCLUDE_TARGET_IN_TOPK = True

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

OPERATIONS_WITH_TARGET = {
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
}


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
    return get_dialogue_id(item), get_speaker(item)


def build_user_fact_index(
    base_fact_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
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

    return fact_id_to_record, user_to_facts


def deterministic_shuffle(items: list[dict[str, Any]], seed_value: str) -> list[dict[str, Any]]:
    items = list(items)
    local_random = random.Random(seed_value)
    local_random.shuffle(items)
    return items


def load_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embedding model: {MODEL_NAME}")
    print(f"Device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    return model


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


def retrieve_top_k(
    model: SentenceTransformer,
    utterance: str,
    facts: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not facts:
        return [], {}

    fact_texts = [fact["fact"] for fact in facts]

    query_embedding = encode_texts(model, [utterance])
    fact_embeddings = encode_texts(model, fact_texts)

    similarities = cosine_similarity(query_embedding, fact_embeddings)[0]

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
    model: SentenceTransformer,
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

    selected_facts, similarity_by_fact_id = retrieve_top_k(
        model=model,
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
        selected_fact_ids = {fact["fact_id"] for fact in selected_facts}

        if fact_id not in selected_fact_ids:
            if len(selected_facts) >= TOP_K:
                selected_facts = selected_facts[:-1]

            selected_facts.append(target_fact_record)

    selected_facts = deterministic_shuffle(
        selected_facts,
        seed_value=f"mpnet_topk::{RANDOM_SEED}::{row_index}::{fact_id}",
    )

    memory_facts = []

    for idx, fact in enumerate(selected_facts, start=1):
        is_gold_target = target_required and fact["fact_id"] == fact_id

        memory_facts.append({
            "prompt_fact_id": idx,
            "original_fact_id": fact["fact_id"],
            "fact": fact["fact"],
            "similarity_to_utterance": similarity_by_fact_id.get(fact["fact_id"]),
            "is_gold_target": is_gold_target,
        })

    target_similarity = similarity_by_fact_id.get(fact_id)

    return {
        "example_id": str(row.get("example_id", "")),
        "source_fact_id": fact_id,
        "base_fact": str(row.get("base_fact", "")),
        "utterance": utterance,
        "gold_operation": operation,
        "context_type": "mpnet_topk_facts",
        "retrieval_method": "mpnet_cosine",
        "embedding_model": MODEL_NAME,
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
    print(f"Retrieval method: {record['retrieval_method']}")
    print()

    print("Memory facts:")
    for fact in record["memory_facts"]:
        score = fact.get("similarity_to_utterance")
        score_text = f" | sim={score:.4f}" if isinstance(score, float) else ""
        marker = "  <-- gold target" if fact.get("is_gold_target") else ""
        print(f"[{fact['prompt_fact_id']}] {fact['fact']}{score_text}{marker}")

    print("\nUtterance:")
    print(record["utterance"])

    print("\nExpected output:")
    print(json.dumps({
        "operation": record["gold_operation"],
    }, ensure_ascii=False, indent=2))


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Loading test split: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)

    print(f"Loading base facts: {BASE_FACTS_PATH}")
    base_fact_records = load_jsonl(BASE_FACTS_PATH)

    fact_id_to_record, user_to_facts = build_user_fact_index(base_fact_records)

    print(f"Base facts loaded: {len(fact_id_to_record)}")
    print(f"Users found: {len(user_to_facts)}")
    print(f"Test examples loaded: {len(test_df)}")

    model = load_embedding_model()

    output_records = []
    skipped = 0

    for row_index, row in test_df.iterrows():
        row_dict = row.to_dict()

        operation = str(row_dict.get("target_operation", "")).strip()

        if operation not in TARGET_OPERATIONS:
            continue

        print(
            f"[{row_index + 1}/{len(test_df)}] "
            f"operation={operation} "
            f"fact_id={row_dict.get('fact_id')}"
        )

        record = make_topk_context(
            row=row_dict,
            fact_id_to_record=fact_id_to_record,
            user_to_facts=user_to_facts,
            model=model,
            row_index=row_index,
        )

        if record is None:
            skipped += 1
            continue

        output_records.append(record)

    save_jsonl(output_records, OUTPUT_PATH)

    update_records = [
        item for item in output_records
        if item["gold_operation"] in OPERATIONS_WITH_TARGET
    ]

    raw_topk_hits = [
        item["target_was_in_raw_topk"]
        for item in update_records
        if item["target_was_in_raw_topk"] is not None
    ]

    retrieval_recall = None

    if raw_topk_hits:
        retrieval_recall = sum(raw_topk_hits) / len(raw_topk_hits)

    report = {
        "input_test_csv": str(TEST_CSV_PATH),
        "input_base_facts": str(BASE_FACTS_PATH),
        "output_path": str(OUTPUT_PATH),
        "model_name": MODEL_NAME,
        "top_k": TOP_K,
        "force_include_target": FORCE_INCLUDE_TARGET_IN_TOPK,
        "total_test_examples": len(test_df),
        "saved_examples": len(output_records),
        "skipped_examples": skipped,
        "update_examples": len(update_records),
        "raw_topk_retrieval_recall_for_target_facts": retrieval_recall,
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"- {OUTPUT_PATH}: {len(output_records)} records")
    print(f"- {REPORT_PATH}")

    print("\nSkipped:")
    print(f"- {skipped}")

    if retrieval_recall is not None:
        print(f"\nRaw top-{TOP_K} retrieval recall for target facts: {retrieval_recall:.4f}")

    if output_records:
        print_example(output_records[0])


if __name__ == "__main__":
    main()