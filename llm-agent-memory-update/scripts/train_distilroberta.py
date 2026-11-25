import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


INPUT_JSONL_PATH = Path("01_dataset_final.jsonl")

OUTPUT_DIR = Path("data/roberta_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV_PATH = OUTPUT_DIR / "roberta_test_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "roberta_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "roberta_confusion_matrix.csv"

MODEL_NAME = "distilroberta-base"

RANDOM_SEED = 42
TEST_SIZE = 0.2

MAX_LENGTH = 192

NUM_TRAIN_EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Если на CPU будет слишком медленно, поставь 4 или 2.
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

LABELS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def make_input_text(base_fact: str, utterance: str) -> str:
    return (
        f"Base fact: {base_fact}\n"
        f"Utterance: {utterance}"
    )


def load_dataset_as_dataframe(path: Path) -> pd.DataFrame:
    records = load_jsonl(path)

    rows = []

    for item in records:
        operation = item.get("target_operation")

        if operation not in label2id:
            continue

        base_fact = str(item.get("base_fact", "")).strip()
        utterance = str(item.get("utterance", "")).strip()
        fact_id = str(item.get("fact_id", "")).strip()

        if not base_fact or not utterance or not fact_id:
            continue

        rows.append({
            "example_id": item.get("example_id"),
            "fact_id": fact_id,
            "base_fact": base_fact,
            "utterance": utterance,
            "text": make_input_text(base_fact, utterance),
            "label_name": operation,
            "label": label2id[operation],
        })

    df = pd.DataFrame(rows)

    return df


def make_group_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df["label"],
            groups=df["fact_id"],
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


def tokenize_dataset(tokenizer, dataset: Dataset) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    return dataset.map(tokenize_batch, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def write_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: list[int],
    y_pred: list[int],
    trainer_metrics: dict[str, Any],
):
    y_true_names = [id2label[int(x)] for x in y_true]
    y_pred_names = [id2label[int(x)] for x in y_pred]

    accuracy = accuracy_score(y_true_names, y_pred_names)
    macro_f1 = f1_score(
        y_true_names,
        y_pred_names,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true_names,
        y_pred_names,
        average="weighted",
        zero_division=0,
    )

    cls_report = classification_report(
        y_true_names,
        y_pred_names,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true_names,
        y_pred_names,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )

    cm_df.to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8-sig")

    lines = []

    lines.append("ROBERTA CLASSIFIER REPORT")
    lines.append("=========================")
    lines.append("")
    lines.append(f"Input file: {INPUT_JSONL_PATH}")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append("Type: fine-tuned neural sequence classifier")
    lines.append(f"Split: GroupShuffleSplit by fact_id")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append(f"Test size: {TEST_SIZE}")
    lines.append("")
    lines.append("Training settings:")
    lines.append(f"- max_length: {MAX_LENGTH}")
    lines.append(f"- num_train_epochs: {NUM_TRAIN_EPOCHS}")
    lines.append(f"- learning_rate: {LEARNING_RATE}")
    lines.append(f"- weight_decay: {WEIGHT_DECAY}")
    lines.append(f"- train_batch_size: {TRAIN_BATCH_SIZE}")
    lines.append(f"- eval_batch_size: {EVAL_BATCH_SIZE}")
    lines.append("")
    lines.append(f"Total examples: {len(train_df) + len(test_df)}")
    lines.append(f"Train examples: {len(train_df)}")
    lines.append(f"Test examples: {len(test_df)}")
    lines.append("")
    lines.append("Overall metrics:")
    lines.append(f"- accuracy: {accuracy:.4f}")
    lines.append(f"- macro_f1: {macro_f1:.4f}")
    lines.append(f"- weighted_f1: {weighted_f1:.4f}")
    lines.append("")
    lines.append("Train distribution:")
    lines.append(train_df["label_name"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Test distribution:")
    lines.append(test_df["label_name"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("Trainer eval metrics:")
    lines.append(json.dumps(trainer_metrics, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("Classification report:")
    lines.append(cls_report)
    lines.append("")
    lines.append("Confusion matrix:")
    lines.append(cm_df.to_string())
    lines.append("")
    lines.append(f"Saved predictions: {PREDICTIONS_CSV_PATH}")
    lines.append(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\nOverall metrics:")
    print(f"- accuracy: {accuracy:.4f}")
    print(f"- macro_f1: {macro_f1:.4f}")
    print(f"- weighted_f1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(cls_report)

    print(f"\nSaved report: {REPORT_PATH}")
    print(f"Saved predictions: {PREDICTIONS_CSV_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    if not INPUT_JSONL_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL_PATH}")

    df = load_dataset_as_dataframe(INPUT_JSONL_PATH)

    print(f"Loaded examples: {len(df)}")
    print(df["label_name"].value_counts().sort_index())

    train_df, test_df = make_group_split(df)

    print(f"\nTrain examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    tokenized_train = tokenize_dataset(tokenizer, train_dataset)
    tokenized_test = tokenize_dataset(tokenizer, test_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=RANDOM_SEED,
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining RoBERTa classifier...")
    trainer.train()

    print("\nEvaluating...")
    trainer_metrics = trainer.evaluate()

    predictions = trainer.predict(tokenized_test)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids

    pred_names = [id2label[int(x)] for x in y_pred]
    true_names = [id2label[int(x)] for x in y_true]

    output_df = test_df.copy()
    output_df["roberta_prediction"] = pred_names
    output_df["is_correct"] = output_df["label_name"] == output_df["roberta_prediction"]

    for label_idx, label_name in id2label.items():
        output_df[f"prob_{label_name}"] = torch.softmax(
            torch.tensor(logits),
            dim=-1,
        ).numpy()[:, label_idx]

    output_df.to_csv(
        PREDICTIONS_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    trainer.save_model(str(OUTPUT_DIR / "best_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))

    write_report(
        train_df=train_df,
        test_df=test_df,
        y_true=list(y_true),
        y_pred=list(y_pred),
        trainer_metrics=trainer_metrics,
    )


if __name__ == "__main__":
    main()