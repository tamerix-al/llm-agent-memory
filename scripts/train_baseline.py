import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


INPUT_PATH = Path("01_dataset_final.jsonl")

OUTPUT_DIR = Path("data/baseline_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "tfidf_logreg_memory_operation_classifier.joblib"
REPORT_PATH = OUTPUT_DIR / "baseline_report.txt"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "test_predictions.csv"
CONFUSION_MATRIX_CSV_PATH = OUTPUT_DIR / "confusion_matrix.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2


TARGET_OPERATIONS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            item["_line_number"] = line_number
            records.append(item)

    return records


def get_utterance(item: dict) -> str:
    utterance = item.get("utterance")

    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()

    parsed_response = item.get("parsed_response")

    if isinstance(parsed_response, dict):
        utterance = parsed_response.get("utterance")

        if isinstance(utterance, str) and utterance.strip():
            return utterance.strip()

    return ""


def get_tags_as_text(item: dict) -> str:
    tags = item.get("tags", [])

    if isinstance(tags, list):
        return ";".join(tags)

    return str(tags)


def make_text_input(item: dict) -> str:
    """
    Даём классификатору и старый факт, и новую реплику.

    Операция памяти определяется отношением:
    base_fact <-> utterance.

    Пример:
    base_fact: I like pizza.
    utterance: I don't like pizza anymore.
    target: DELETE

    base_fact: I like pizza.
    utterance: I don't like pizza.
    target: CONFLICT
    """
    base_fact = str(item.get("base_fact", "")).strip()
    utterance = get_utterance(item)

    return f"Base fact: {base_fact}\nUtterance: {utterance}"


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []

    for item in records:
        operation = item.get("target_operation")
        utterance = get_utterance(item)

        if operation not in TARGET_OPERATIONS:
            continue

        if not utterance:
            continue

        rows.append({
            "line_number": item.get("_line_number", ""),
            "example_id": item.get("example_id", ""),
            "fact_id": item.get("fact_id", ""),
            "base_fact": item.get("base_fact", ""),
            "utterance": utterance,
            "text": make_text_input(item),
            "target_operation": operation,
            "tags": get_tags_as_text(item),
            "prompt_version": item.get("prompt_version", ""),
            "model": item.get("model", ""),
        })

    return pd.DataFrame(rows)


def main():
    print(f"Loading dataset: {INPUT_PATH}")

    records = load_jsonl(INPUT_PATH)
    df = records_to_dataframe(records)

    print(f"Loaded valid examples: {len(df)}")

    if len(df) == 0:
        raise ValueError("No valid examples found.")

    print("\nClass distribution:")
    print(df["target_operation"].value_counts().sort_index())

    X = df["text"]
    y = df["target_operation"]
    groups = df["fact_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(f"\nTrain examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["target_operation"].value_counts().sort_index())

    print("\nTest distribution:")
    print(test_df["target_operation"].value_counts().sort_index())

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                solver="lbfgs",
            )
        ),
    ])

    print("\nTraining model...")
    pipeline.fit(train_df["text"], train_df["target_operation"])

    print("Predicting...")
    y_pred = pipeline.predict(test_df["text"])

    accuracy = accuracy_score(test_df["target_operation"], y_pred)
    macro_f1 = f1_score(
        test_df["target_operation"],
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        test_df["target_operation"],
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        test_df["target_operation"],
        y_pred,
        labels=TARGET_OPERATIONS,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        test_df["target_operation"],
        y_pred,
        labels=TARGET_OPERATIONS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in TARGET_OPERATIONS],
        columns=[f"pred_{label}" for label in TARGET_OPERATIONS],
    )

    test_df["prediction"] = y_pred
    test_df["is_correct"] = test_df["target_operation"] == test_df["prediction"]

    test_df.to_csv(PREDICTIONS_CSV_PATH, index=False, encoding="utf-8-sig")
    cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8-sig")

    joblib.dump(pipeline, MODEL_PATH)

    report_text = []
    report_text.append("BASELINE CLASSIFIER REPORT")
    report_text.append("==========================")
    report_text.append("")
    report_text.append(f"Input file: {INPUT_PATH}")
    report_text.append("Model: TF-IDF + Logistic Regression")
    report_text.append("Split: GroupShuffleSplit by fact_id")
    report_text.append(f"Random seed: {RANDOM_SEED}")
    report_text.append(f"Test size: {TEST_SIZE}")
    report_text.append("")
    report_text.append(f"Total examples: {len(df)}")
    report_text.append(f"Train examples: {len(train_df)}")
    report_text.append(f"Test examples: {len(test_df)}")
    report_text.append("")
    report_text.append("Overall metrics:")
    report_text.append(f"- accuracy: {accuracy:.4f}")
    report_text.append(f"- macro_f1: {macro_f1:.4f}")
    report_text.append(f"- weighted_f1: {weighted_f1:.4f}")
    report_text.append("")
    report_text.append("Class distribution:")
    report_text.append(df["target_operation"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Train distribution:")
    report_text.append(train_df["target_operation"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Test distribution:")
    report_text.append(test_df["target_operation"].value_counts().sort_index().to_string())
    report_text.append("")
    report_text.append("Classification report:")
    report_text.append(report)
    report_text.append("")
    report_text.append("Confusion matrix:")
    report_text.append(cm_df.to_string())
    report_text.append("")
    report_text.append(f"Saved model: {MODEL_PATH}")
    report_text.append(f"Saved predictions: {PREDICTIONS_CSV_PATH}")
    report_text.append(f"Saved confusion matrix: {CONFUSION_MATRIX_CSV_PATH}")

    REPORT_PATH.write_text("\n".join(report_text), encoding="utf-8")

    print("\nOverall metrics:")
    print(f"- accuracy: {accuracy:.4f}")
    print(f"- macro_f1: {macro_f1:.4f}")
    print(f"- weighted_f1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm_df)

    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved predictions: {PREDICTIONS_CSV_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_CSV_PATH}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()