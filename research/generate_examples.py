import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_FACTS_PATH = Path("data/processed/base_facts_upd.jsonl")
PROMPT_PATH = Path("prompts/prompt_one_tag2.txt")

OUTPUT_PATH = Path("data/generated/single_operation_test_v2.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "qwen/qwen3.5-122b-a10b"

# Сколько base facts взять для теста.
# Каждый base fact даст 6 generated examples.
N_BASE_FACTS = 500

TEMPERATURE = 0.8
TOP_P = 0.9
MAX_TOKENS = 300

REASONING_EFFORT = "none"

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

TAGS_BY_OPERATION = {
    "ADD": ["ADD"],
    "DELETE": ["DELETE"],
    "DELETE_ADD": ["DELETE", "ADD"],
    "CONFLICT": ["CONFLICT"],
    "IGNORE-NONFACT": ["IGNORE-NONFACT"],
    "IGNORE-OTHERS": ["IGNORE-OTHERS"],
}


def load_jsonl(path: Path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def extract_json_from_response(text: str):
    """
    Пытается распарсить ответ модели как JSON.
    Если модель обернула JSON в markdown-блок, убирает обёртку.
    """
    raw = text.strip()

    if raw.startswith("```json"):
        raw = raw.removeprefix("```json").strip()
        raw = raw.removesuffix("```").strip()
    elif raw.startswith("```"):
        raw = raw.removeprefix("```").strip()
        raw = raw.removesuffix("```").strip()

    return json.loads(raw)


def load_existing_example_keys(path: Path):
    """
    Читает уже существующий output-файл и возвращает пары:
    (fact_id, target_operation)

    Это нужно, чтобы при повторном запуске скрипт не генерировал
    те же самые операции для тех же самых фактов.
    """
    existing_keys = set()

    if not path.exists():
        return existing_keys

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            fact_id = item.get("fact_id")
            target_operation = item.get("target_operation")

            if fact_id and target_operation:
                existing_keys.add((fact_id, target_operation))

    return existing_keys


def count_existing_examples(path: Path):
    if not path.exists():
        return 0

    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def build_prompt(prompt_template: str, base_fact: str, target_operation: str):
    prompt = prompt_template.replace("{BASE_FACT}", base_fact)
    prompt = prompt.replace("{TARGET_OPERATION}", target_operation)
    return prompt


def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Check your .env file.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    base_facts = load_jsonl(BASE_FACTS_PATH)[:N_BASE_FACTS]

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    existing_keys = load_existing_example_keys(OUTPUT_PATH)
    existing_count = count_existing_examples(OUTPUT_PATH)

    print(f"Total selected base facts: {len(base_facts)}")
    print(f"Target operations: {len(TARGET_OPERATIONS)}")
    print(f"Expected max new examples: {len(base_facts) * len(TARGET_OPERATIONS)}")
    print(f"Already existing examples in output: {existing_count}")
    print(f"Model: {MODEL}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    print(f"Output: {OUTPUT_PATH}")

    example_counter = existing_count

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for fact_record in base_facts:
            fact_id = fact_record["fact_id"]
            base_fact = fact_record["base_fact"]

            for target_operation in TARGET_OPERATIONS:
                key = (fact_id, target_operation)

                if key in existing_keys:
                    print(f"Skipping existing: {fact_id} | {target_operation}")
                    continue

                example_counter += 1
                example_id = f"ex_{example_counter:06d}"

                prompt = build_prompt(
                    prompt_template=prompt_template,
                    base_fact=base_fact,
                    target_operation=target_operation
                )

                result = {
                    "example_id": example_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "parameters": {
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_tokens": MAX_TOKENS,
                        "reasoning_effort": REASONING_EFFORT
                    },
                    "prompt_version": "single_operation_v1",
                    "fact_id": fact_id,
                    "base_fact": base_fact,
                    "base_fact_record": fact_record,
                    "target_operation": target_operation,

                    # Теги задаются программно, а не моделью.
                    "tags": TAGS_BY_OPERATION[target_operation],

                    "raw_prompt": prompt,
                    "raw_response": None,
                    "utterance": None,
                    "parsed_response": None,
                    "parse_status": "not_started",
                    "usage": None,
                    "error": None
                }

                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS,
                        extra_body={
                            "reasoning": {
                                "effort": REASONING_EFFORT
                            }
                        }
                    )

                    if response.usage:
                        result["usage"] = response.usage.model_dump()

                    raw_response = response.choices[0].message.content
                    result["raw_response"] = raw_response

                    try:
                        parsed = extract_json_from_response(raw_response)
                        result["parsed_response"] = parsed

                        utterance = parsed.get("utterance")

                        if isinstance(utterance, str) and utterance.strip():
                            result["utterance"] = utterance.strip()
                            result["parse_status"] = "ok"
                        else:
                            result["parse_status"] = "missing_utterance"
                            result["error"] = "Parsed JSON does not contain a valid utterance."

                    except Exception as parse_error:
                        result["parse_status"] = "parse_error"
                        result["error"] = f"Parse error: {parse_error}"

                except Exception as api_error:
                    result["parse_status"] = "api_error"
                    result["error"] = f"API error: {api_error}"

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()

                print(
                    f"{example_id} | {result['parse_status']} | "
                    f"{fact_id} | {target_operation} | {base_fact}"
                )

                existing_keys.add(key)

                time.sleep(0.5)

    print("Done.")
    print(f"Saved/appended to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()