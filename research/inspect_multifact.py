import json
from pathlib import Path
from collections import defaultdict


DATASET_PATH = Path("01_dataset_final.jsonl")

# Лучше взять DELETE или DELETE_ADD, потому что там target_fact_id наглядный.
PREFERRED_OPERATIONS = ["DELETE", "DELETE_ADD", "CONFLICT"]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    return records


def get_utterance(item: dict) -> str:
    utterance = item.get("utterance")

    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()

    parsed = item.get("parsed_response")

    if isinstance(parsed, dict):
        utterance = parsed.get("utterance")
        if isinstance(utterance, str):
            return utterance.strip()

    return ""


def get_base_fact_record(item: dict) -> dict:
    record = item.get("base_fact_record")

    if isinstance(record, dict):
        return record

    return {}


def get_user_key(item: dict):
    """
    Группируем факты одного пользователя.

    Обычно у тебя в base_fact_record должны быть:
    dialogue_id
    speaker

    Если вдруг их нет, fallback будет менее полезным.
    """
    record = get_base_fact_record(item)

    dialogue_id = record.get("dialogue_id")
    speaker = record.get("speaker")

    if dialogue_id is None:
        dialogue_id = item.get("dialogue_id")

    if speaker is None:
        speaker = item.get("speaker")

    return str(dialogue_id), str(speaker)


def get_fact_id(item: dict) -> str:
    return str(item.get("fact_id", ""))


def get_base_fact(item: dict) -> str:
    return str(item.get("base_fact", "")).strip()


def main():
    records = load_jsonl(DATASET_PATH)

    print(f"Loaded records: {len(records)}")

    user_to_facts = defaultdict(dict)

    for item in records:
        user_key = get_user_key(item)
        fact_id = get_fact_id(item)
        base_fact = get_base_fact(item)

        if fact_id and base_fact:
            user_to_facts[user_key][fact_id] = base_fact

    chosen = None
    chosen_context = None

    for operation in PREFERRED_OPERATIONS:
        for item in records:
            if item.get("target_operation") != operation:
                continue

            user_key = get_user_key(item)
            context_facts_dict = user_to_facts.get(user_key, {})

            # Хотим пример, где есть хотя бы 4 факта в контексте.
            if len(context_facts_dict) >= 4:
                chosen = item
                chosen_context = context_facts_dict
                break

        if chosen is not None:
            break

    if chosen is None:
        print("Could not find a suitable example with enough context facts.")
        return

    target_fact_id = get_fact_id(chosen)
    target_fact = get_base_fact(chosen)
    utterance = get_utterance(chosen)
    operation = chosen.get("target_operation")

    context_items = list(chosen_context.items())

    # Чтобы пример был не огромный, покажем максимум 8 фактов.
    # Но target fact обязательно оставим.
    max_facts_to_show = 8

    selected_context = []

    for fact_id, fact_text in context_items:
        if fact_id == target_fact_id:
            selected_context.append((fact_id, fact_text))

    for fact_id, fact_text in context_items:
        if fact_id == target_fact_id:
            continue
        selected_context.append((fact_id, fact_text))

        if len(selected_context) >= max_facts_to_show:
            break

    # Перенумеруем для prompt-а: [1], [2], [3]...
    prompt_facts = []
    target_prompt_id = None

    for idx, (fact_id, fact_text) in enumerate(selected_context, start=1):
        prompt_facts.append({
            "prompt_fact_id": idx,
            "original_fact_id": fact_id,
            "fact": fact_text,
            "is_target": fact_id == target_fact_id,
        })

        if fact_id == target_fact_id:
            target_prompt_id = idx

    print("\nPAIR SETUP")
    print("==========")
    print("Base fact:")
    print(target_fact)
    print("\nUtterance:")
    print(utterance)
    print("\nGold operation:")
    print(operation)

    print("\nMULTI-FACT SETUP")
    print("================")

    print("\nInput:")
    print("Memory facts:")

    for fact in prompt_facts:
        print(f"[{fact['prompt_fact_id']}] {fact['fact']}")

    print("\nUtterance:")
    print(utterance)

    print("\nExpected output:")
    expected_output = {
        "operation": operation,
        "target_fact_id": target_prompt_id,
        "target_fact": target_fact,
    }

    print(json.dumps(expected_output, ensure_ascii=False, indent=2))

    print("\nDEBUG METADATA")
    print("==============")
    print(f"original target fact_id: {target_fact_id}")
    print(f"user key: {get_user_key(chosen)}")
    print(f"available facts for this user: {len(chosen_context)}")


if __name__ == "__main__":
    main()