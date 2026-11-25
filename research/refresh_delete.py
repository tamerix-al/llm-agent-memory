import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


INPUT_PATH = Path("data/final/memory_operations_cleaned_2.jsonl")
PROMPT_PATH = Path("prompts/delete_2.txt")

OUTPUT_DIR = Path("data/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "memory_operations_delete_refreshed_style_controlled.jsonl"
REPORT_PATH = OUTPUT_DIR / "delete_style_controlled_refresh_report.json"

MODEL = "qwen/qwen3.5-122b-a10b"

TEMPERATURE = 0.9
TOP_P = 0.95
MAX_TOKENS = 300
REASONING_EFFORT = "none"

RANDOM_SEED = 42

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

# Сколько старых DELETE оставить без изменений.
# Если у тебя 435 DELETE, то:
# 150 старых останутся
# 285 будут перегенерированы
KEEP_OLD_DELETE = 150

MAX_RETRIES_PER_EXAMPLE = 3
SLEEP_SECONDS = 0.2


DELETE_STYLES = [
    "direct_negation",
    "stopped",
    "no_longer",
    "used_to_but_not_now",
    "gave_up_or_quit",
    "not_true_now",
    "soft_correction",
    "status_ended",
]


# Эти паттерны запрещены, потому что они часто превращают DELETE в DELETE_ADD.
FORBIDDEN_DELETE_PATTERNS = [
    " instead",
    "switched to",
    "started ",
    "began ",
    "now i prefer",
    "now i usually",
    "now i have",
    "now i work",
    "now i live",
    "became ",
    "moved to",
    "adopted ",
    "bought ",
    "purchased ",
    "signed up",
    "joined ",
]


# Эти паттерны не запрещены, но мы считаем их частотность.
DELETE_SURFACE_PATTERNS = {
    "dont_anymore": [
        "don't",
        "anymore",
    ],
    "do_not_anymore": [
        "do not",
        "anymore",
    ],
    "not_anymore": [
        "not",
        "anymore",
    ],
    "no_longer": [
        "no longer",
    ],
    "stopped": [
        "stopped",
    ],
    "used_to": [
        "used to",
    ],
    "these_days": [
        "these days",
    ],
    "gave_up": [
        "gave up",
    ],
    "quit": [
        "quit",
    ],
    "not_true_now": [
        "not true",
        "now",
    ],
    "outdated": [
        "outdated",
    ],
    "would_not_describe": [
        "wouldn't describe",
    ],
    "would_not_say": [
        "wouldn't say",
    ],
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


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            cleaned = dict(item)
            cleaned.pop("_source_line_number", None)
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


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


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text


def contains_forbidden_delete_pattern(text: str) -> list[str]:
    text_norm = normalize_text(text)
    hits = []

    for pattern in FORBIDDEN_DELETE_PATTERNS:
        if pattern in text_norm:
            hits.append(pattern.strip())

    return hits


def detect_surface_pattern(text: str) -> str:
    text_norm = normalize_text(text)

    for pattern_name, required_parts in DELETE_SURFACE_PATTERNS.items():
        if all(part in text_norm for part in required_parts):
            return pattern_name

    return "other"


def detect_all_surface_patterns(text: str) -> list[str]:
    text_norm = normalize_text(text)
    patterns = []

    for pattern_name, required_parts in DELETE_SURFACE_PATTERNS.items():
        if all(part in text_norm for part in required_parts):
            patterns.append(pattern_name)

    if not patterns:
        patterns.append("other")

    return patterns


def parse_model_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()

    # На случай если модель обернула JSON в markdown.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    return json.loads(raw_text)


def make_prompt(prompt_template: str, base_fact: str, delete_style: str) -> str:
    prompt = prompt_template.replace("{BASE_FACT}", base_fact)
    prompt = prompt.replace("{DELETE_STYLE}", delete_style)
    return prompt


def call_model(client: OpenAI, prompt: str) -> tuple[str, dict | None]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        extra_body={
            "reasoning": {
                "effort": REASONING_EFFORT
            }
        },
    )

    raw_text = response.choices[0].message.content

    usage = None

    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return raw_text, usage


def assign_delete_styles(n: int) -> list[str]:
    """
    Создаёт список стилей длиной n.

    Стили распределяются максимально ровно:
    если нужно 285 примеров и 8 стилей,
    каждый стиль получит примерно 35-36 примеров.
    """
    styles = []

    while len(styles) < n:
        styles.extend(DELETE_STYLES)

    styles = styles[:n]

    random.seed(RANDOM_SEED)
    random.shuffle(styles)

    return styles


def style_was_followed(utterance: str, delete_style: str) -> bool:
    """
    Мягкая эвристика.

    Мы не используем это как жёсткий фильтр,
    потому что иногда fallback может быть нормальным.
    Но сохраняем в отчёт.
    """
    text = normalize_text(utterance)

    if delete_style == "direct_negation":
        return (
            "don't" in text
            or "do not" in text
            or "i'm not" in text
            or "i am not" in text
            or "not" in text
        )

    if delete_style == "stopped":
        return "stopped" in text

    if delete_style == "no_longer":
        return "no longer" in text

    if delete_style == "used_to_but_not_now":
        return "used to" in text

    if delete_style == "gave_up_or_quit":
        return "gave up" in text or "quit" in text

    if delete_style == "not_true_now":
        return (
            "not true" in text
            or "isn't true" in text
            or "is not true" in text
            or "doesn't describe" in text
            or "does not describe" in text
            or "used to be true" in text
        )

    if delete_style == "soft_correction":
        return (
            "wouldn't describe" in text
            or "would not describe" in text
            or "wouldn't say" in text
            or "would not say" in text
            or "outdated" in text
            or "doesn't describe" in text
            or "does not describe" in text
        )

    if delete_style == "status_ended":
        return (
            "not" in text
            or "no longer" in text
            or "don't have" in text
            or "do not have" in text
            or "don't live" in text
            or "do not live" in text
        )

    return False


def create_refreshed_delete_example(
    client: OpenAI,
    prompt_template: str,
    old_item: dict,
    refresh_index: int,
    delete_style: str,
) -> tuple[dict, dict]:
    base_fact = old_item.get("base_fact", "")
    prompt = make_prompt(prompt_template, base_fact, delete_style)

    attempts = []

    for attempt_number in range(1, MAX_RETRIES_PER_EXAMPLE + 1):
        try:
            raw_response, usage = call_model(client, prompt)
            parsed = parse_model_json(raw_response)

            utterance = parsed.get("utterance", "")

            if not isinstance(utterance, str) or not utterance.strip():
                attempts.append({
                    "attempt": attempt_number,
                    "status": "bad_response",
                    "reason": "missing_or_empty_utterance",
                    "raw_response": raw_response,
                })
                continue

            utterance = utterance.strip()
            forbidden_hits = contains_forbidden_delete_pattern(utterance)

            if forbidden_hits:
                attempts.append({
                    "attempt": attempt_number,
                    "status": "rejected",
                    "reason": "forbidden_delete_pattern",
                    "forbidden_hits": forbidden_hits,
                    "utterance": utterance,
                    "raw_response": raw_response,
                })
                continue

            followed_style = style_was_followed(utterance, delete_style)
            surface_pattern = detect_surface_pattern(utterance)
            all_surface_patterns = detect_all_surface_patterns(utterance)

            new_item = dict(old_item)

            new_item["example_id"] = (
                f"{old_item.get('example_id', '')}"
                f"_delete_style_refresh_{refresh_index:04d}"
            )
            new_item["target_operation"] = "DELETE"
            new_item["tags"] = ["DELETE"]
            new_item["utterance"] = utterance
            new_item["parsed_response"] = parsed
            new_item["raw_response"] = raw_response
            new_item["raw_prompt"] = prompt
            new_item["parse_status"] = "ok"
            new_item["error"] = None
            new_item["model"] = MODEL
            new_item["prompt_version"] = "prompt_delete_style_controlled_v1"
            new_item["created_at"] = datetime.now(timezone.utc).isoformat()
            new_item["parameters"] = {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "reasoning_effort": REASONING_EFFORT,
            }
            new_item["usage"] = usage
            new_item["refresh_metadata"] = {
                "source_example_id": old_item.get("example_id", ""),
                "source_prompt_version": old_item.get("prompt_version", ""),
                "source_utterance": get_utterance(old_item),
                "refresh_reason": "increase_delete_surface_diversity",
                "required_delete_style": delete_style,
                "style_was_followed_heuristic": followed_style,
                "surface_pattern": surface_pattern,
                "all_surface_patterns": all_surface_patterns,
                "attempt_number": attempt_number,
            }

            attempts.append({
                "attempt": attempt_number,
                "status": "accepted",
                "utterance": utterance,
                "required_delete_style": delete_style,
                "style_was_followed_heuristic": followed_style,
                "surface_pattern": surface_pattern,
                "all_surface_patterns": all_surface_patterns,
            })

            report_item = {
                "source_example_id": old_item.get("example_id", ""),
                "new_example_id": new_item["example_id"],
                "fact_id": old_item.get("fact_id", ""),
                "base_fact": base_fact,
                "required_delete_style": delete_style,
                "style_was_followed_heuristic": followed_style,
                "old_utterance": get_utterance(old_item),
                "new_utterance": utterance,
                "surface_pattern": surface_pattern,
                "all_surface_patterns": all_surface_patterns,
                "attempts": attempts,
                "status": "ok",
            }

            return new_item, report_item

        except Exception as e:
            attempts.append({
                "attempt": attempt_number,
                "status": "error",
                "error": repr(e),
            })

        time.sleep(SLEEP_SECONDS)

    # Если все попытки провалились, возвращаем старый пример,
    # чтобы не ломать баланс.
    fallback_item = dict(old_item)
    fallback_item["refresh_metadata"] = {
        "refresh_reason": "fallback_old_delete_after_failed_style_controlled_refresh",
        "required_delete_style": delete_style,
        "style_was_followed_heuristic": False,
        "surface_pattern": detect_surface_pattern(get_utterance(old_item)),
        "all_surface_patterns": detect_all_surface_patterns(get_utterance(old_item)),
    }

    report_item = {
        "source_example_id": old_item.get("example_id", ""),
        "new_example_id": old_item.get("example_id", ""),
        "fact_id": old_item.get("fact_id", ""),
        "base_fact": base_fact,
        "required_delete_style": delete_style,
        "style_was_followed_heuristic": False,
        "old_utterance": get_utterance(old_item),
        "new_utterance": get_utterance(old_item),
        "surface_pattern": detect_surface_pattern(get_utterance(old_item)),
        "all_surface_patterns": detect_all_surface_patterns(get_utterance(old_item)),
        "attempts": attempts,
        "status": "fallback_old_item",
    }

    return fallback_item, report_item


def count_by_operation(records: list[dict]) -> dict[str, int]:
    counts = {}

    for operation in TARGET_OPERATIONS:
        counts[operation] = sum(
            1 for item in records
            if item.get("target_operation") == operation
        )

    return counts


def count_delete_surface_patterns(records: list[dict]) -> dict[str, int]:
    counter = Counter()

    delete_records = [
        item for item in records
        if item.get("target_operation") == "DELETE"
    ]

    for item in delete_records:
        pattern = detect_surface_pattern(get_utterance(item))
        counter[pattern] += 1

    return dict(counter.most_common())


def count_required_styles(records: list[dict]) -> dict[str, int]:
    counter = Counter()

    for item in records:
        if item.get("target_operation") != "DELETE":
            continue

        metadata = item.get("refresh_metadata", {})

        if not isinstance(metadata, dict):
            continue

        style = metadata.get("required_delete_style")

        if style:
            counter[style] += 1

    return dict(counter.most_common())


def count_style_following(report_items: list[dict]) -> dict:
    total = 0
    followed = 0
    by_style = {}

    for item in report_items:
        if item.get("status") != "ok":
            continue

        style = item.get("required_delete_style", "UNKNOWN")
        was_followed = bool(item.get("style_was_followed_heuristic"))

        total += 1

        if was_followed:
            followed += 1

        if style not in by_style:
            by_style[style] = {
                "total": 0,
                "followed": 0,
                "not_followed": 0,
            }

        by_style[style]["total"] += 1

        if was_followed:
            by_style[style]["followed"] += 1
        else:
            by_style[style]["not_followed"] += 1

    return {
        "total_ok_refreshes": total,
        "style_followed_heuristic": followed,
        "style_not_followed_heuristic": total - followed,
        "style_following_rate": round(followed / total, 4) if total else None,
        "by_style": by_style,
    }


def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    print(f"Loading dataset: {INPUT_PATH}")

    records = load_jsonl(INPUT_PATH)

    print(f"Loaded records: {len(records)}")

    non_delete_records = [
        item for item in records
        if item.get("target_operation") != "DELETE"
    ]

    delete_records = [
        item for item in records
        if item.get("target_operation") == "DELETE"
    ]

    print(f"DELETE records found: {len(delete_records)}")

    if KEEP_OLD_DELETE >= len(delete_records):
        raise ValueError(
            f"KEEP_OLD_DELETE={KEEP_OLD_DELETE} must be smaller than DELETE count={len(delete_records)}"
        )

    random.seed(RANDOM_SEED)

    shuffled_delete = delete_records.copy()
    random.shuffle(shuffled_delete)

    old_delete_to_keep = shuffled_delete[:KEEP_OLD_DELETE]
    old_delete_to_refresh = shuffled_delete[KEEP_OLD_DELETE:]

    delete_styles = assign_delete_styles(len(old_delete_to_refresh))

    print(f"Keeping old DELETE examples: {len(old_delete_to_keep)}")
    print(f"Refreshing DELETE examples: {len(old_delete_to_refresh)}")

    print("\nAssigned style distribution:")
    assigned_style_counts = Counter(delete_styles)

    for style in DELETE_STYLES:
        print(f"- {style}: {assigned_style_counts[style]}")

    refreshed_delete_records = []
    refresh_report_items = []

    for index, (old_item, delete_style) in enumerate(
        zip(old_delete_to_refresh, delete_styles),
        start=1,
    ):
        print(
            f"[{index}/{len(old_delete_to_refresh)}] "
            f"Refreshing DELETE for {old_item.get('fact_id', '')} "
            f"with style={delete_style}"
        )

        new_item, report_item = create_refreshed_delete_example(
            client=client,
            prompt_template=prompt_template,
            old_item=old_item,
            refresh_index=index,
            delete_style=delete_style,
        )

        refreshed_delete_records.append(new_item)
        refresh_report_items.append(report_item)

        time.sleep(SLEEP_SECONDS)

    final_records = non_delete_records + old_delete_to_keep + refreshed_delete_records

    operation_order = {
        operation: index
        for index, operation in enumerate(TARGET_OPERATIONS)
    }

    final_records.sort(
        key=lambda item: (
            str(item.get("fact_id", "")),
            operation_order.get(item.get("target_operation", ""), 999),
            str(item.get("example_id", "")),
        )
    )

    write_jsonl(final_records, OUTPUT_PATH)

    report = {
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_PATH),
        "prompt_file": str(PROMPT_PATH),
        "model": MODEL,
        "random_seed": RANDOM_SEED,

        "settings": {
            "keep_old_delete": KEEP_OLD_DELETE,
            "refreshed_delete_target_count": len(old_delete_to_refresh),
            "delete_styles": DELETE_STYLES,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "max_retries_per_example": MAX_RETRIES_PER_EXAMPLE,
        },

        "final_examples": len(final_records),
        "examples_per_class": count_by_operation(final_records),

        "delete_surface_patterns_before": count_delete_surface_patterns(records),
        "delete_surface_patterns_after": count_delete_surface_patterns(final_records),

        "assigned_delete_styles_for_refreshed_examples": dict(assigned_style_counts),
        "required_delete_styles_in_final_dataset": count_required_styles(final_records),

        "style_following_summary": count_style_following(refresh_report_items),

        "refresh_items": refresh_report_items,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved refreshed dataset: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")

    print("\nFinal class distribution:")
    for operation, count in count_by_operation(final_records).items():
        print(f"- {operation}: {count}")

    print("\nDELETE surface patterns before:")
    for pattern, count in count_delete_surface_patterns(records).items():
        print(f"- {pattern}: {count}")

    print("\nDELETE surface patterns after:")
    for pattern, count in count_delete_surface_patterns(final_records).items():
        print(f"- {pattern}: {count}")

    print("\nStyle-following summary:")
    style_summary = count_style_following(refresh_report_items)
    print(
        f"- followed: {style_summary['style_followed_heuristic']} / "
        f"{style_summary['total_ok_refreshes']}"
    )
    print(f"- rate: {style_summary['style_following_rate']}")

    print("\nDone.")


if __name__ == "__main__":
    main()