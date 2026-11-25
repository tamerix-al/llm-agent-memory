import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


INPUT_CSV_PATH = Path("data/baseline_classifier/test_predictions.csv")

OUTPUT_DIR = Path("data/qwen_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV_PATH = OUTPUT_DIR / "qwen_classifier_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "qwen_classifier_report.txt"
CONFUSION_MATRIX_CSV_PATH = OUTPUT_DIR / "qwen_confusion_matrix.csv"
RAW_OUTPUTS_JSONL_PATH = OUTPUT_DIR / "qwen_raw_outputs.jsonl"

MODEL = "qwen/qwen3.5-122b-a10b"

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 80
REASONING_EFFORT = "none"

SLEEP_SECONDS = 0.2
MAX_RETRIES = 3

TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

LABEL_SET = set(TARGET_OPERATIONS)


PROMPT_TEMPLATE = """
You are a classifier for a conversational memory system.

Your task is to classify the relationship between an existing user memory and a new user utterance.

Return exactly one label.

Allowed labels:
ADD
DELETE
DELETE_ADD
CONFLICT
IGNORE-NONFACT
IGNORE-OTHERS

Definitions:

ADD:
The utterance introduces a new stable personal fact about the speaker.
It is compatible with the base fact.
It does not delete, modify, or contradict the base fact.

DELETE:
The utterance shows that the base fact is no longer true.
It does not introduce a new replacement fact.
Examples of deletion signals include: anymore, no longer, stopped, quit, gave up, used to but not now.

DELETE_ADD:
The utterance shows that the base fact is no longer true and also introduces a new replacement or related personal fact.

CONFLICT:
The utterance contradicts the base fact, but does not clearly present it as a change over time.
There is no clear update signal such as anymore, no longer, stopped, moved, quit, started, switched, or used to.

IGNORE-NONFACT:
The utterance does not contain a stable personal fact about the speaker.
It may be a question, reaction, filler, vague comment, or conversational acknowledgement.

IGNORE-OTHERS:
The utterance contains a fact about someone other than the speaker.
It does not introduce a personal fact about the speaker.

Important distinctions:

"I don't like pizza." may be CONFLICT if the base fact is "I like pizza."
"I don't like pizza anymore." is DELETE.

"I stopped playing guitar." is DELETE.
"I stopped playing guitar and started painting." is DELETE_ADD.

"My friend plays guitar." is IGNORE-OTHERS.
"That sounds interesting." is IGNORE-NONFACT.

Input:

Base fact:
{BASE_FACT}

Utterance:
{UTTERANCE}

Return only valid JSON in this exact format:

{{"label": "..."}}
""".strip()


def normalize_label(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    text = text.strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            label = parsed.get("label")

            if isinstance(label, str):
                label = label.strip()

                if label in LABEL_SET:
                    return label

    except Exception:
        pass

    # fallback на случай, если модель вернула не JSON, а просто label
    upper_text = text.upper()

    for label in TARGET_OPERATIONS:
        if re.search(rf"\b{re.escape(label)}\b", upper_text):
            return label

    return None


def make_prompt(base_fact: str, utterance: str) -> str:
    return PROMPT_TEMPLATE.replace("{BASE_FACT}", str(base_fact)).replace(
        "{UTTERANCE}",
        str(utterance),
    )


def call_qwen(client: OpenAI, prompt: str) -> tuple[str, dict | None]:
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


def classify_example(
    client: OpenAI,
    row: dict,
    row_index: int,
) -> tuple[str | None, dict]:
    base_fact = row.get("base_fact", "")
    utterance = row.get("utterance", "")
    prompt = make_prompt(base_fact, utterance)

    attempts = []

    for attempt_number in range(1, MAX_RETRIES + 1):
        try:
            raw_output, usage = call_qwen(client, prompt)
            predicted_label = normalize_label(raw_output)

            attempt = {
                "attempt": attempt_number,
                "raw_output": raw_output,
                "parsed_label": predicted_label,
                "usage": usage,
                "status": "ok" if predicted_label is not None else "parse_failed",
            }

            attempts.append(attempt)

            if predicted_label is not None:
                raw_record = {
                    "row_index": row_index,
                    "example_id": row.get("example_id", ""),
                    "fact_id": row.get("fact_id", ""),
                    "base_fact": base_fact,
                    "utterance": utterance,
                    "gold_label": row.get("target_operation", ""),
                    "predicted_label": predicted_label,
                    "attempts": attempts,
                    "final_status": "ok",
                }

                return predicted_label, raw_record

        except Exception as e:
            attempts.append({
                "attempt": attempt_number,
                "status": "error",
                "error": repr(e),
            })

        time.sleep(SLEEP_SECONDS)

    raw_record = {
        "row_index": row_index,
        "example_id": row.get("example_id", ""),
        "fact_id": row.get("fact_id", ""),
        "base_fact": base_fact,
        "utterance": utterance,
        "gold_label": row.get("target_operation", ""),
        "predicted_label": None,
        "attempts": attempts,
        "final_status": "failed",
    }

    return None, raw_record


def save_raw_outputs(raw_records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in raw_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env")

    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV_PATH}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    print(f"Loading test set: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    required_columns = [
        "base_fact",
        "utterance",
        "target_operation",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in CSV: {column}")

    print(f"Loaded examples: {len(df)}")

    predictions = []
    raw_records = []

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        print(
            f"[{i + 1}/{len(df)}] "
            f"gold={row_dict.get('target_operation', '')} "
            f"example_id={row_dict.get('example_id', '')}"
        )

        predicted_label, raw_record = classify_example(
            client=client,
            row=row_dict,
            row_index=i,
        )

        predictions.append(predicted_label)
        raw_records.append(raw_record)

        time.sleep(SLEEP_SECONDS)

    df["qwen_prediction"] = predictions
    df["qwen_parse_ok"] = df["qwen_prediction"].notna()
    df["qwen_is_correct"] = df["target_operation"] == df["qwen_prediction"]

    df.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="utf-8-sig")
    save_raw_outputs(raw_records, RAW_OUTPUTS_JSONL_PATH)

    eval_df = df[df["qwen_parse_ok"]].copy()

    failed_count = len(df) - len(eval_df)

    if len(eval_df) == 0:
        raise ValueError("No successfully parsed Qwen predictions.")

    y_true = eval_df["target_operation"]
    y_pred = eval_df["qwen_prediction"]

    accuracy = accuracy_score(y_true, y_pred)

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

    cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8-sig")

    report_text = []
    report_text.append("QWEN PROMPTED CLASSIFIER REPORT")
    report_text.append("================================")
    report_text.append("")
    report_text.append(f"Input file: {INPUT_CSV_PATH}")
    report_text.append(f"Model: {MODEL}")
    report_text.append("Type: prompted LLM classifier")
    report_text.append("Test set: same test split as TF-IDF baseline")
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
    report_text.append("Overall metrics:")
    report_text.append(f"- accuracy: {accuracy:.4f}")
    report_text.append(f"- macro_f1: {macro_f1:.4f}")
    report_text.append(f"- weighted_f1: {weighted_f1:.4f}")
    report_text.append("")
    report_text.append("Gold class distribution:")
    report_text.append(df["target_operation"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Predicted class distribution:")
    report_text.append(eval_df["qwen_prediction"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Classification report:")
    report_text.append(report)
    report_text.append("")
    report_text.append("Confusion matrix:")
    report_text.append(cm_df.to_string())
    report_text.append("")
    report_text.append(f"Saved predictions: {PREDICTIONS_CSV_PATH}")
    report_text.append(f"Saved confusion matrix: {CONFUSION_MATRIX_CSV_PATH}")
    report_text.append(f"Saved raw outputs: {RAW_OUTPUTS_JSONL_PATH}")

    REPORT_PATH.write_text("\n".join(report_text), encoding="utf-8")

    print("\nOverall metrics:")
    print(f"- accuracy: {accuracy:.4f}")
    print(f"- macro_f1: {macro_f1:.4f}")
    print(f"- weighted_f1: {weighted_f1:.4f}")
    print(f"- failed predictions: {failed_count}")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm_df)

    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved predictions: {PREDICTIONS_CSV_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_CSV_PATH}")
    print(f"Saved raw outputs: {RAW_OUTPUTS_JSONL_PATH}")


if __name__ == "__main__":
    main()