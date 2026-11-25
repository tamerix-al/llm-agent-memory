import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# Меняй этот путь, чтобы прогнать другой setup:
# 1) data/multifact_context/multifact_all_user_facts.jsonl
# 2) data/multifact_context/multifact_retrieved_topk_facts.jsonl
INPUT_JSONL_PATH = Path("data/multifact_context/multifact_all_user_facts.jsonl")

MODEL = "qwen/qwen3.5-122b-a10b"

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 120
REASONING_EFFORT = "none"

SLEEP_SECONDS = 0.2
MAX_RETRIES = 3

# Для тестового запуска можешь поставить 50.
# Для полного запуска поставь None.
MAX_EXAMPLES = None

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

LABEL_SET = set(TARGET_OPERATIONS)

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


PROMPT_TEMPLATE = """
You are a classifier for a conversational memory system.

You are given:
1. A numbered list of existing memory facts about the user.
2. A new user utterance.

Your task is to predict the memory operation for the new utterance.

Return exactly one operation.

Allowed operations:
ADD
DELETE
DELETE_ADD
CONFLICT
IGNORE-NONFACT
IGNORE-OTHERS

Definitions:

ADD:
The utterance introduces a new stable personal fact about the speaker.
It is compatible with the existing memory facts.
It does not delete, modify, or contradict any existing memory fact.

DELETE:
The utterance shows that an existing memory fact is no longer true.
It does not introduce a new replacement fact.
Examples of deletion signals include: anymore, no longer, stopped, quit, gave up, used to but not now.

DELETE_ADD:
The utterance shows that an existing memory fact is no longer true and also introduces a new replacement or related personal fact.

CONFLICT:
The utterance contradicts an existing memory fact, but does not clearly present it as a change over time.
There is no clear update signal such as anymore, no longer, stopped, moved, quit, started, switched, or used to.

IGNORE-NONFACT:
The utterance does not contain a stable personal fact about the speaker.
It may be a question, reaction, filler, vague comment, or conversational acknowledgement.

IGNORE-OTHERS:
The utterance contains a fact about someone other than the speaker.
It does not introduce a personal fact about the speaker.

Important distinctions:

"I don't like pizza." may be CONFLICT if there is an existing memory fact "I like pizza."
"I don't like pizza anymore." is DELETE for the memory fact "I like pizza."

"I stopped playing guitar." is DELETE.
"I stopped playing guitar and started painting." is DELETE_ADD.

"My friend plays guitar." is IGNORE-OTHERS.
"That sounds interesting." is IGNORE-NONFACT.

Rules:
- Use the memory facts as context.
- Predict only the operation.
- Do not return a target fact id.
- Return only valid JSON.
- Do not explain your answer.

Memory facts:
{MEMORY_FACTS}

Utterance:
{UTTERANCE}

Return exactly this JSON format:

{{
  "operation": "..."
}}
""".strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def append_jsonl(item: dict[str, Any], path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_output_paths(input_path: Path) -> dict[str, Path]:
    stem = input_path.stem

    output_dir = Path("data/qwen_multifact_classifier") / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "predictions_csv": output_dir / "qwen_multifact_predictions.csv",
        "report": output_dir / "qwen_multifact_report.txt",
        "confusion_matrix_csv": output_dir / "qwen_multifact_confusion_matrix.csv",
        "raw_outputs_jsonl": output_dir / "qwen_multifact_raw_outputs.jsonl",
    }


def format_memory_facts(memory_facts: list[dict[str, Any]]) -> str:
    lines = []

    for fact in memory_facts:
        prompt_fact_id = fact.get("prompt_fact_id")
        fact_text = fact.get("fact", "")

        lines.append(f"[{prompt_fact_id}] {fact_text}")

    return "\n".join(lines)


def make_prompt(item: dict[str, Any]) -> str:
    memory_facts_text = format_memory_facts(item.get("memory_facts", []))
    utterance = str(item.get("utterance", ""))

    return (
        PROMPT_TEMPLATE
        .replace("{MEMORY_FACTS}", memory_facts_text)
        .replace("{UTTERANCE}", utterance)
    )


def normalize_operation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip()

    if value in LABEL_SET:
        return value

    upper_value = value.upper()

    for label in TARGET_OPERATIONS:
        if upper_value == label.upper():
            return label

    return None


def normalize_target_fact_id(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"null", "none", "no", "n/a", "na", ""}:
            return None

        text = text.replace("[", "").replace("]", "")

        if text.isdigit():
            return int(text)

    return None


def parse_qwen_output(text: str) -> tuple[str | None, int | None, dict[str, Any] | None]:
    """
    Ожидаем JSON:
    {
      "operation": "DELETE",
      "target_fact_id": 2
    }

    Но добавляем fallback, если модель вернула неидеальный формат.
    """
    if not isinstance(text, str):
        return None, None, None

    raw_text = text.strip()

    try:
        parsed = json.loads(raw_text)

        if isinstance(parsed, dict):
            operation = normalize_operation(
                parsed.get("operation") or parsed.get("label")
            )

            target_fact_id = normalize_target_fact_id(
                parsed.get("target_fact_id")
            )

            return operation, target_fact_id, parsed

    except Exception:
        pass

    # Fallback: пробуем вытащить JSON-подстроку
    json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))

            if isinstance(parsed, dict):
                operation = normalize_operation(
                    parsed.get("operation") or parsed.get("label")
                )

                target_fact_id = normalize_target_fact_id(
                    parsed.get("target_fact_id")
                )

                return operation, target_fact_id, parsed

        except Exception:
            pass

    # Fallback: ищем operation в тексте
    operation = None

    for label in TARGET_OPERATIONS:
        if re.search(rf"\b{re.escape(label)}\b", raw_text, flags=re.IGNORECASE):
            operation = label
            break

    target_fact_id = None

    target_match = re.search(
        r"target[_\s-]*fact[_\s-]*id[^0-9]*(\d+)",
        raw_text,
        flags=re.IGNORECASE,
    )

    if target_match:
        target_fact_id = int(target_match.group(1))

    return operation, target_fact_id, None


def call_qwen(client: OpenAI, prompt: str) -> tuple[str, dict[str, Any] | None]:
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


def get_gold_target_fact_id(item: dict[str, Any]) -> int | None:
    return normalize_target_fact_id(item.get("gold_target_fact_id"))


def classify_example(
    client: OpenAI,
    item: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    prompt = make_prompt(item)

    attempts = []

    for attempt_number in range(1, MAX_RETRIES + 1):
        try:
            raw_output, usage = call_qwen(client, prompt)

            pred_operation, pred_target_fact_id, parsed_json = parse_qwen_output(raw_output)

            status = "ok" if pred_operation is not None else "parse_failed"

            attempt = {
                "attempt": attempt_number,
                "status": status,
                "raw_output": raw_output,
                "parsed_json": parsed_json,
                "pred_operation": pred_operation,
                "pred_target_fact_id": pred_target_fact_id,
                "usage": usage,
            }

            attempts.append(attempt)

            if pred_operation is not None:
                return {
                    "row_index": row_index,
                    "example_id": item.get("example_id"),
                    "context_type": item.get("context_type"),
                    "gold_operation": item.get("gold_operation"),
                    "gold_target_fact_id": get_gold_target_fact_id(item),
                    "gold_target_fact": item.get("gold_target_fact"),
                    "pred_operation": pred_operation,
                    "pred_target_fact_id": pred_target_fact_id,
                    "final_status": "ok",
                    "attempts": attempts,
                }

        except Exception as e:
            attempts.append({
                "attempt": attempt_number,
                "status": "error",
                "error": repr(e),
            })

        time.sleep(SLEEP_SECONDS)

    return {
        "row_index": row_index,
        "example_id": item.get("example_id"),
        "context_type": item.get("context_type"),
        "gold_operation": item.get("gold_operation"),
        "gold_target_fact_id": get_gold_target_fact_id(item),
        "gold_target_fact": item.get("gold_target_fact"),
        "pred_operation": None,
        "pred_target_fact_id": None,
        "final_status": "failed",
        "attempts": attempts,
    }


def is_target_correct(row: pd.Series) -> bool:
    gold = normalize_target_fact_id(row.get("gold_target_fact_id"))
    pred = normalize_target_fact_id(row.get("pred_target_fact_id"))

    return gold == pred


def make_prediction_rows(
    items: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    raw_by_index = {
        int(record["row_index"]): record
        for record in raw_records
    }

    for idx, item in enumerate(items):
        raw_record = raw_by_index.get(idx, {})

        memory_facts = item.get("memory_facts", [])

        rows.append({
            "row_index": idx,
            "example_id": item.get("example_id"),
            "context_type": item.get("context_type"),
            "source_fact_id": item.get("source_fact_id"),
            "base_fact": item.get("base_fact"),
            "utterance": item.get("utterance"),
            "num_memory_facts": item.get("num_memory_facts"),
            "gold_operation": item.get("gold_operation"),
            "gold_target_fact_id": get_gold_target_fact_id(item),
            "gold_target_fact": item.get("gold_target_fact"),
            "pred_operation": raw_record.get("pred_operation"),
            "pred_target_fact_id": raw_record.get("pred_target_fact_id"),
            "parse_ok": raw_record.get("final_status") == "ok",
            "operation_correct": item.get("gold_operation") == raw_record.get("pred_operation"),
            "target_correct": normalize_target_fact_id(item.get("gold_target_fact_id")) == normalize_target_fact_id(raw_record.get("pred_target_fact_id")),
            "memory_facts_text": format_memory_facts(memory_facts),
        })

    df = pd.DataFrame(rows)

    df["joint_correct"] = df["operation_correct"] & df["target_correct"]

    return df


def write_report(
    df: pd.DataFrame,
    paths: dict[str, Path],
):
    eval_df = df[df["parse_ok"]].copy()

    failed_count = len(df) - len(eval_df)

    if len(eval_df) == 0:
        raise ValueError("No successfully parsed predictions.")

    y_true = eval_df["gold_operation"]
    y_pred = eval_df["pred_operation"]

    operation_accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_OPERATIONS,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=TARGET_OPERATIONS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in TARGET_OPERATIONS],
        columns=[f"pred_{label}" for label in TARGET_OPERATIONS],
    )

    cm_df.to_csv(paths["confusion_matrix_csv"], encoding="utf-8-sig")

    target_required_df = eval_df[
        eval_df["gold_operation"].isin(OPERATIONS_WITH_TARGET)
    ].copy()

    target_not_required_df = eval_df[
        eval_df["gold_operation"].isin(OPERATIONS_WITHOUT_TARGET)
    ].copy()

    target_accuracy_all = float(eval_df["target_correct"].mean())
    joint_accuracy_all = float(eval_df["joint_correct"].mean())

    if len(target_required_df) > 0:
        target_accuracy_required = float(target_required_df["target_correct"].mean())
        joint_accuracy_required = float(target_required_df["joint_correct"].mean())
    else:
        target_accuracy_required = None
        joint_accuracy_required = None

    if len(target_not_required_df) > 0:
        null_target_accuracy = float(target_not_required_df["target_correct"].mean())
    else:
        null_target_accuracy = None

    report_text = []

    report_text.append("QWEN MULTI-FACT CLASSIFIER REPORT")
    report_text.append("=================================")
    report_text.append("")
    report_text.append(f"Input file: {INPUT_JSONL_PATH}")
    report_text.append(f"Model: {MODEL}")
    report_text.append("Type: prompted LLM classifier")
    report_text.append("")
    report_text.append("Generation settings:")
    report_text.append(f"- temperature: {TEMPERATURE}")
    report_text.append(f"- top_p: {TOP_P}")
    report_text.append(f"- max_tokens: {MAX_TOKENS}")
    report_text.append(f"- reasoning_effort: {REASONING_EFFORT}")
    report_text.append("")
    report_text.append(f"Total examples: {len(df)}")
    report_text.append(f"Parsed predictions: {len(eval_df)}")
    report_text.append(f"Failed predictions: {failed_count}")
    report_text.append("")
    report_text.append("Operation metrics:")
    report_text.append(f"- operation_accuracy: {operation_accuracy:.4f}")
    report_text.append(f"- operation_macro_f1: {macro_f1:.4f}")
    report_text.append(f"- operation_weighted_f1: {weighted_f1:.4f}")
    report_text.append("")
    report_text.append("Target fact metrics:")
    report_text.append(f"- target_accuracy_all: {target_accuracy_all:.4f}")
    report_text.append(f"- joint_accuracy_all: {joint_accuracy_all:.4f}")

    if target_accuracy_required is not None:
        report_text.append(f"- target_accuracy_required_DELETE_DELETE_ADD_CONFLICT: {target_accuracy_required:.4f}")
        report_text.append(f"- joint_accuracy_required_DELETE_DELETE_ADD_CONFLICT: {joint_accuracy_required:.4f}")

    if null_target_accuracy is not None:
        report_text.append(f"- null_target_accuracy_ADD_IGNORE: {null_target_accuracy:.4f}")

    report_text.append("")
    report_text.append("Gold operation distribution:")
    report_text.append(eval_df["gold_operation"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Predicted operation distribution:")
    report_text.append(eval_df["pred_operation"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Classification report:")
    report_text.append(report)
    report_text.append("")
    report_text.append("Confusion matrix:")
    report_text.append(cm_df.to_string())
    report_text.append("")
    report_text.append(f"Saved predictions: {paths['predictions_csv']}")
    report_text.append(f"Saved confusion matrix: {paths['confusion_matrix_csv']}")
    report_text.append(f"Saved raw outputs: {paths['raw_outputs_jsonl']}")

    paths["report"].write_text("\n".join(report_text), encoding="utf-8")

    print("\nOperation metrics:")
    print(f"- operation_accuracy: {operation_accuracy:.4f}")
    print(f"- operation_macro_f1: {macro_f1:.4f}")
    print(f"- operation_weighted_f1: {weighted_f1:.4f}")

    print("\nTarget fact metrics:")
    print(f"- target_accuracy_all: {target_accuracy_all:.4f}")
    print(f"- joint_accuracy_all: {joint_accuracy_all:.4f}")

    if target_accuracy_required is not None:
        print(f"- target_accuracy_required_DELETE_DELETE_ADD_CONFLICT: {target_accuracy_required:.4f}")
        print(f"- joint_accuracy_required_DELETE_DELETE_ADD_CONFLICT: {joint_accuracy_required:.4f}")

    if null_target_accuracy is not None:
        print(f"- null_target_accuracy_ADD_IGNORE: {null_target_accuracy:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm_df)

    print(f"\nSaved report: {paths['report']}")
    print(f"Saved predictions: {paths['predictions_csv']}")
    print(f"Saved confusion matrix: {paths['confusion_matrix_csv']}")
    print(f"Saved raw outputs: {paths['raw_outputs_jsonl']}")


def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")

    if not INPUT_JSONL_PATH.exists():
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL_PATH}")

    paths = get_output_paths(INPUT_JSONL_PATH)

    # Чтобы случайно не смешать старые raw outputs с новым запуском.
    if paths["raw_outputs_jsonl"].exists():
        print(f"Removing old raw output file: {paths['raw_outputs_jsonl']}")
        paths["raw_outputs_jsonl"].unlink()

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    items = load_jsonl(INPUT_JSONL_PATH)

    if MAX_EXAMPLES is not None:
        items = items[:MAX_EXAMPLES]

    print(f"Loaded examples: {len(items)}")
    print(f"Input: {INPUT_JSONL_PATH}")
    print(f"Output directory: {paths['output_dir']}")

    raw_records = []

    for idx, item in enumerate(items):
        print(
            f"[{idx + 1}/{len(items)}] "
            f"context={item.get('context_type')} "
            f"gold_operation={item.get('gold_operation')} "
            f"example_id={item.get('example_id')}"
        )

        raw_record = classify_example(
            client=client,
            item=item,
            row_index=idx,
        )

        raw_records.append(raw_record)
        append_jsonl(raw_record, paths["raw_outputs_jsonl"])

        time.sleep(SLEEP_SECONDS)

    predictions_df = make_prediction_rows(
        items=items,
        raw_records=raw_records,
    )

    predictions_df.to_csv(
        paths["predictions_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    write_report(
        df=predictions_df,
        paths=paths,
    )


if __name__ == "__main__":
    main()