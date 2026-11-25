import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = Path("data/roberta_multifact_classifier/final_model")
INPUT_PATH = Path("data/manual_eval/my_examples.jsonl")

OUTPUT_DIR = Path("data/manual_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "manual_eval_predictions.csv"
REPORT_TXT = OUTPUT_DIR / "manual_eval_report.txt"
CONFUSION_CSV = OUTPUT_DIR / "manual_eval_confusion_matrix.csv"

LABELS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON error at line {line_no}: {e}")

    return rows


def normalize_memory_facts(memory_facts):
    facts = []

    for item in memory_facts:
        if isinstance(item, str):
            fact = item
        elif isinstance(item, dict):
            fact = item.get("fact", "")
        else:
            fact = str(item)

        fact = fact.strip()

        if fact:
            facts.append(fact)

    return facts


def format_input(row: dict) -> str:
    facts = normalize_memory_facts(row.get("memory_facts", []))

    fact_lines = []
    for i, fact in enumerate(facts, start=1):
        fact_lines.append(f"{i}. {fact}")

    facts_text = "\n".join(fact_lines)
    utterance = str(row.get("utterance", "")).strip()

    return (
        "Memory facts:\n"
        f"{facts_text}\n\n"
        "User utterance:\n"
        f"{utterance}\n\n"
        "Task: classify the memory operation."
    )


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Сначала нужно сохранить обученную модель в final_model."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Manual eval file not found: {INPUT_PATH}")

    rows = load_jsonl(INPUT_PATH)

    if not rows:
        raise ValueError("Manual eval file is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Input file: {INPUT_PATH}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    output_rows = []

    for row in rows:
        example_id = row.get("example_id")
        expected = row.get("expected_operation")
        utterance = str(row.get("utterance", "")).strip()
        facts = normalize_memory_facts(row.get("memory_facts", []))

        if not utterance:
            raise ValueError(f"Empty utterance in example: {example_id}")

        if expected not in LABELS:
            raise ValueError(f"Bad expected_operation in {example_id}: {expected}")

        if not facts:
            raise ValueError(f"No memory_facts in example: {example_id}")

        text = format_input(row)

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        pred_id = int(probs.argmax())
        predicted = LABELS[pred_id]
        confidence = float(probs[pred_id])

        result = {
            "example_id": example_id,
            "expected_operation": expected,
            "predicted_operation": predicted,
            "is_correct": expected == predicted,
            "confidence": confidence,
            "utterance": utterance,
            "memory_facts": " | ".join(facts),
        }

        for label, prob in zip(LABELS, probs):
            result[f"prob_{label}"] = float(prob)

        output_rows.append(result)

        print(
            f"{example_id}: expected={expected}, "
            f"predicted={predicted}, "
            f"correct={expected == predicted}, "
            f"confidence={confidence:.4f}"
        )

    df = pd.DataFrame(output_rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    y_true = df["expected_operation"]
    y_pred = df["predicted_operation"]

    accuracy = accuracy_score(y_true, y_pred)

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    cm_df.to_csv(CONFUSION_CSV, encoding="utf-8-sig")

    report_lines = []
    report_lines.append("MANUAL OUT-OF-DATASET EVALUATION REPORT")
    report_lines.append("=======================================")
    report_lines.append("")
    report_lines.append(f"Model: {MODEL_DIR}")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Total examples: {len(df)}")
    report_lines.append("")
    report_lines.append(f"Accuracy: {accuracy:.4f}")
    report_lines.append("")
    report_lines.append("Classification report:")
    report_lines.append(cls_report)
    report_lines.append("")
    report_lines.append("Confusion matrix:")
    report_lines.append(cm_df.to_string())
    report_lines.append("")
    report_lines.append("Errors:")

    errors = df[df["is_correct"] == False]

    if len(errors) == 0:
        report_lines.append("No errors.")
    else:
        report_lines.append(
            errors[
                [
                    "example_id",
                    "expected_operation",
                    "predicted_operation",
                    "confidence",
                    "utterance",
                ]
            ].to_string(index=False)
        )

    report_text = "\n".join(report_lines)
    REPORT_TXT.write_text(report_text, encoding="utf-8")

    print()
    print(report_text)
    print()
    print(f"Saved predictions: {OUTPUT_CSV}")
    print(f"Saved report: {REPORT_TXT}")
    print(f"Saved confusion matrix: {CONFUSION_CSV}")


if __name__ == "__main__":
    main()