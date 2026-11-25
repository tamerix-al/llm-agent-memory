import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "nayohan/multi_session_chat"

OUTPUT_JSONL = Path("data/processed/base_facts.jsonl")
OUTPUT_CSV = Path("data/processed/base_facts_preview.csv")

# Сколько фактов максимум взять.
# Поставь None, если хочешь выгрузить все.
MAX_FACTS = 750

EXISTING_JSONL = Path("data/processed/base_facts.jsonl")
TARGET_TOTAL_FACTS = 1000

# Из каких split брать факты.
SPLITS = ["test"]


def normalize_fact(text: str) -> str:
    """
    Минимальная очистка факта.
    Не переписываем смысл, только убираем лишние пробелы.
    """
    text = str(text).strip()
    text = " ".join(text.split())
    return text


def get_dialogue_id(row: dict):
    """
    В датасете поле называется dialoug_id с опечаткой.
    На всякий случай поддерживаем оба варианта.
    """
    if "dialoug_id" in row:
        return row["dialoug_id"]
    if "dialogue_id" in row:
        return row["dialogue_id"]
    return None




def load_existing_records(path: Path):
    if not path.exists():
        return [], set()

    records = []
    seen_keys = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            records.append(item)

            key = (
                str(item.get("dialogue_id")),
                str(item.get("session_id")),
                str(item.get("speaker")),
                str(item.get("original_text")).strip().lower()
            )
            seen_keys.add(key)

    return records, seen_keys


def add_persona_facts(records, row, split_name, persona_field, speaker_name, seen_keys):
    persona = row.get(persona_field)

    if persona is None:
        return

    dialogue_id = get_dialogue_id(row)
    session_id = row.get("session_id")

    for fact in persona:
        original_text = str(fact).strip()
        base_fact = normalize_fact(original_text)

        if not base_fact:
            continue

        key = (
            str(dialogue_id),
            str(session_id),
            str(speaker_name),
            original_text.strip().lower()
        )

        if key in seen_keys:
            continue

        record = {
            "fact_id": f"bf_{len(records) + 1:06d}",
            "base_fact": base_fact,
            "source_dataset": DATASET_NAME,
            "split": split_name,
            "dialogue_id": dialogue_id,
            "session_id": session_id,
            "speaker": speaker_name,
            "fact_source": "persona",
            "original_text": original_text,
            "was_edited": base_fact != original_text,
            "notes": ""
        }

        records.append(record)
        seen_keys.add(key)

        if len(records) >= TARGET_TOTAL_FACTS:
            return


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    records, seen_keys = load_existing_records(EXISTING_JSONL)

    print(f"Existing facts: {len(records)}")
    print(f"Target total facts: {TARGET_TOTAL_FACTS}")

    if len(records) >= TARGET_TOTAL_FACTS:
        print("Already reached target. Nothing to add.")
        return

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    print("Available splits:", list(dataset.keys()))

    for split_name in SPLITS:
        if split_name not in dataset:
            print(f"Split not found, skipping: {split_name}")
            continue

        print(f"Processing split: {split_name}")

        for row in dataset[split_name]:
            add_persona_facts(
                records=records,
                row=row,
                split_name=split_name,
                persona_field="persona1",
                speaker_name="Speaker 1",
                seen_keys=seen_keys
            )

            if len(records) >= TARGET_TOTAL_FACTS:
                break

            add_persona_facts(
                records=records,
                row=row,
                split_name=split_name,
                persona_field="persona2",
                speaker_name="Speaker 2",
                seen_keys=seen_keys
            )

            if len(records) >= TARGET_TOTAL_FACTS:
                break

        if len(records) >= TARGET_TOTAL_FACTS:
            break

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved JSONL: {OUTPUT_JSONL}")
    print(f"Saved CSV preview: {OUTPUT_CSV}")
    print(f"Total facts: {len(records)}")

    if records:
        print("\nLast record:")
        print(json.dumps(records[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()