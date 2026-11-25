import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


INPUT_PATH = Path("data/multifact_training/multifact_all_user_facts_from_final.jsonl")

OUTPUT_DIR = Path("data/roberta_multifact_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "roberta_multifact_from_final_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "roberta_multifact_from_final_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "roberta_multifact_from_final_confusion_matrix.csv"

MODEL_NAME = "distilroberta-base"

LABELS = [
    "ADD",
    "DELETE",
    "DELETE_ADD",
    "CONFLICT",
    "IGNORE-NONFACT",
    "IGNORE-OTHERS",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

RANDOM_SEED = 42
TEST_SIZE = 0.2

MAX_LENGTH = 256
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def format_input(row: dict) -> str:
    memory_facts = row.get("memory_facts", [])

    fact_lines = []

    for i, item in enumerate(memory_facts, start=1):
        fact = item.get("fact", "")
        fact_lines.append(f"{i}. {fact}")

    facts_text = "\n".join(fact_lines)

    utterance = row.get("utterance", "")

    return (
        "Memory facts:\n"
        f"{facts_text}\n\n"
        "User utterance:\n"
        f"{utterance}\n\n"
        "Task: classify the memory operation."
    )


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )

        encoded["labels"] = int(self.labels[idx])

        return encoded


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def make_training_arguments():
    kwargs = {
        "output_dir": str(OUTPUT_DIR / "checkpoints"),
        "logging_strategy": "steps",
        "logging_steps": 20,
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "save_strategy": "no",
        "load_best_model_at_end": False,
        "report_to": "none",
        "seed": RANDOM_SEED,
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if torch.cuda.is_available():
        kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def main():
    set_seed(RANDOM_SEED)

    rows = load_jsonl(INPUT_PATH)

    df = pd.DataFrame(rows)

    df["label_name"] = df["gold_operation"]
    df["label_id"] = df["label_name"].map(LABEL2ID)
    df["text"] = df.apply(format_input, axis=1)

    if df["label_id"].isna().any():
        unknown_labels = df[df["label_id"].isna()]["label_name"].unique()
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    if "user_key" not in df.columns:
        raise ValueError("Expected user_key column in multi-fact dataset.")

    groups = df["user_key"].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df["label_id"],
            groups=groups,
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    user_overlap = len(
        set(train_df["user_key"].astype(str))
        & set(test_df["user_key"].astype(str))
    )

    print(f"Loaded examples: {len(df)}")
    print(f"Train examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"User-level overlap: {user_overlap}")
    print()
    print("Train distribution:")
    print(train_df["label_name"].value_counts().sort_index())
    print()
    print("Test distribution:")
    print(test_df["label_name"].value_counts().sort_index())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextClassificationDataset(
        texts=train_df["text"],
        labels=train_df["label_id"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    test_dataset = TextClassificationDataset(
        texts=test_df["text"],
        labels=test_df["label_id"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = make_training_arguments()

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    try:
        trainer = Trainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            **trainer_kwargs,
            tokenizer=tokenizer,
        )

    print("\nTraining started...")
    trainer.train()

    print("\nEvaluating...")
    eval_metrics = trainer.evaluate()

    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)

    test_df["pred_label_id"] = pred_ids
    test_df["pred_operation"] = test_df["pred_label_id"].map(ID2LABEL)
    test_df["is_correct"] = test_df["pred_operation"] == test_df["label_name"]

    output_columns = [
        "example_id",
        "fact_id",
        "user_key",
        "num_memory_facts",
        "base_fact",
        "utterance",
        "label_name",
        "pred_operation",
        "is_correct",
        "text",
    ]

    available_output_columns = [
        col for col in output_columns
        if col in test_df.columns
    ]

    test_df[available_output_columns].to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    y_true = test_df["label_name"].tolist()
    y_pred = test_df["pred_operation"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )

    cm_df.to_csv(
        CONFUSION_MATRIX_PATH,
        encoding="utf-8-sig",
    )

    train_dist = train_df["label_name"].value_counts().sort_index()
    test_dist = test_df["label_name"].value_counts().sort_index()

    report_lines = []

    report_lines.append("ROBERTA MULTI-FACT CLASSIFIER REPORT")
    report_lines.append("====================================")
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Model: {MODEL_NAME}")
    report_lines.append("Type: fine-tuned neural sequence classifier")
    report_lines.append("Input setup: memory facts list + utterance → operation")
    report_lines.append("Split: GroupShuffleSplit by user_key")
    report_lines.append(f"Random seed: {RANDOM_SEED}")
    report_lines.append(f"Test size: {TEST_SIZE}")
    report_lines.append("")
    report_lines.append("Training settings:")
    report_lines.append(f"- max_length: {MAX_LENGTH}")
    report_lines.append(f"- num_train_epochs: {NUM_TRAIN_EPOCHS}")
    report_lines.append(f"- learning_rate: {LEARNING_RATE}")
    report_lines.append(f"- weight_decay: {WEIGHT_DECAY}")
    report_lines.append(f"- train_batch_size: {TRAIN_BATCH_SIZE}")
    report_lines.append(f"- eval_batch_size: {EVAL_BATCH_SIZE}")
    report_lines.append("")
    report_lines.append(f"Total examples: {len(df)}")
    report_lines.append(f"Train examples: {len(train_df)}")
    report_lines.append(f"Test examples: {len(test_df)}")
    report_lines.append(f"User-level overlap between train and test: {user_overlap}")
    report_lines.append("")
    report_lines.append("Overall metrics:")
    report_lines.append(f"- accuracy: {accuracy:.4f}")
    report_lines.append(f"- macro_f1: {macro_f1:.4f}")
    report_lines.append(f"- weighted_f1: {weighted_f1:.4f}")
    report_lines.append("")
    report_lines.append("Train distribution:")
    report_lines.append(train_dist.to_string())
    report_lines.append("")
    report_lines.append("Test distribution:")
    report_lines.append(test_dist.to_string())
    report_lines.append("")
    report_lines.append("Trainer eval metrics:")
    report_lines.append(json.dumps(eval_metrics, ensure_ascii=False, indent=2))
    report_lines.append("")
    report_lines.append("Classification report:")
    report_lines.append(cls_report)
    report_lines.append("")
    report_lines.append("Confusion matrix:")
    report_lines.append(cm_df.to_string())
    report_lines.append("")
    report_lines.append(f"Saved predictions: {PREDICTIONS_PATH}")
    report_lines.append(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")

    report_text = "\n".join(report_lines)

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    MODEL_SAVE_DIR = OUTPUT_DIR / "final_model"
    trainer.save_model(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))

    print(f"Saved model to: {MODEL_SAVE_DIR}")

    print()
    print(report_text)


if __name__ == "__main__":
    main()
