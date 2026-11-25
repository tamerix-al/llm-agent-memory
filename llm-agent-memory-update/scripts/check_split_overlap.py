import json
import re
from pathlib import Path

import pandas as pd


FINAL_DATASET_PATH = Path("01_dataset_final.jsonl")
TEST_PREDICTIONS_PATH = Path("data/baseline_classifier/test_predictions.csv")

OUTPUT_DIR = Path("data/leakage_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "leakage_report.txt"
OVERLAP_BASE_FACTS_PATH = OUTPUT_DIR / "overlap_base_facts.csv"
OVERLAP_UTTERANCES_PATH = OUTPUT_DIR / "overlap_utterances.csv"
OVERLAP_INPUTS_PATH = OUTPUT_DIR / "overlap_inputs.csv"
OVERLAP_USERS_PATH = OUTPUT_DIR / "overlap_users.csv"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_dialogue_id(row):
    # Some records store metadata inside base_fact_record
    if "dialogue_id" in row and row["dialogue_id"]:
        return row["dialogue_id"]
    if "dialog_id" in row and row["dialog_id"]:
        return row["dialog_id"]

    rec = row.get("base_fact_record", {})
    if isinstance(rec, dict):
        return (
            rec.get("dialogue_id")
            or rec.get("dialog_id")
            or rec.get("dialoug_id")
            or rec.get("dialoug_id")
        )

    return None


def get_speaker(row):
    if "speaker" in row and row["speaker"]:
        return row["speaker"]

    rec = row.get("base_fact_record", {})
    if isinstance(rec, dict):
        return rec.get("speaker")

    return None


def main():
    all_rows = load_jsonl(FINAL_DATASET_PATH)
    all_df = pd.DataFrame(all_rows)

    test_df = pd.read_csv(TEST_PREDICTIONS_PATH)

    # Try to detect the example id column in test predictions
    if "example_id" not in test_df.columns:
        raise ValueError("test_predictions.csv must contain an example_id column.")

    test_ids = set(test_df["example_id"].astype(str))
    all_df["example_id"] = all_df["example_id"].astype(str)

    all_df["split"] = all_df["example_id"].apply(lambda x: "test" if x in test_ids else "train")

    train_df = all_df[all_df["split"] == "train"].copy()
    test_full_df = all_df[all_df["split"] == "test"].copy()

    # Normalize fields
    for df in [train_df, test_full_df]:
        df["norm_base_fact"] = df["base_fact"].apply(normalize_text)
        df["norm_utterance"] = df["utterance"].apply(normalize_text)
        df["norm_input"] = df["norm_base_fact"] + " [SEP] " + df["norm_utterance"]
        df["dialogue_id"] = df.apply(get_dialogue_id, axis=1)
        df["speaker"] = df.apply(get_speaker, axis=1)
        df["user_key"] = df["dialogue_id"].astype(str) + "||" + df["speaker"].astype(str)

    # Basic split sizes
    total_examples = len(all_df)
    train_examples = len(train_df)
    test_examples = len(test_full_df)

    # Overlaps
    train_example_ids = set(train_df["example_id"])
    test_example_ids = set(test_full_df["example_id"])

    train_fact_ids = set(train_df["fact_id"].astype(str))
    test_fact_ids = set(test_full_df["fact_id"].astype(str))

    train_base_facts = set(train_df["norm_base_fact"])
    test_base_facts = set(test_full_df["norm_base_fact"])

    train_utterances = set(train_df["norm_utterance"])
    test_utterances = set(test_full_df["norm_utterance"])

    train_inputs = set(train_df["norm_input"])
    test_inputs = set(test_full_df["norm_input"])

    train_users = set(train_df["user_key"])
    test_users = set(test_full_df["user_key"])

    overlap_example_ids = train_example_ids & test_example_ids
    overlap_fact_ids = train_fact_ids & test_fact_ids
    overlap_base_facts = train_base_facts & test_base_facts
    overlap_utterances = train_utterances & test_utterances
    overlap_inputs = train_inputs & test_inputs
    overlap_users = train_users & test_users

    # Save overlap examples for inspection
    if overlap_base_facts:
        out = test_full_df[test_full_df["norm_base_fact"].isin(overlap_base_facts)][
            ["example_id", "fact_id", "base_fact", "utterance", "target_operation", "dialogue_id", "speaker"]
        ]
        out.to_csv(OVERLAP_BASE_FACTS_PATH, index=False, encoding="utf-8-sig")

    if overlap_utterances:
        out = test_full_df[test_full_df["norm_utterance"].isin(overlap_utterances)][
            ["example_id", "fact_id", "base_fact", "utterance", "target_operation", "dialogue_id", "speaker"]
        ]
        out.to_csv(OVERLAP_UTTERANCES_PATH, index=False, encoding="utf-8-sig")

    if overlap_inputs:
        out = test_full_df[test_full_df["norm_input"].isin(overlap_inputs)][
            ["example_id", "fact_id", "base_fact", "utterance", "target_operation", "dialogue_id", "speaker"]
        ]
        out.to_csv(OVERLAP_INPUTS_PATH, index=False, encoding="utf-8-sig")

    if overlap_users:
        out = test_full_df[test_full_df["user_key"].isin(overlap_users)][
            ["example_id", "fact_id", "base_fact", "utterance", "target_operation", "dialogue_id", "speaker", "user_key"]
        ]
        out.to_csv(OVERLAP_USERS_PATH, index=False, encoding="utf-8-sig")

    report_lines = []
    report_lines.append("DATA LEAKAGE CHECK REPORT")
    report_lines.append("=========================")
    report_lines.append("")
    report_lines.append(f"Final dataset: {FINAL_DATASET_PATH}")
    report_lines.append(f"Test predictions file: {TEST_PREDICTIONS_PATH}")
    report_lines.append("")
    report_lines.append("Split reconstruction:")
    report_lines.append(f"- total examples: {total_examples}")
    report_lines.append(f"- train examples: {train_examples}")
    report_lines.append(f"- test examples: {test_examples}")
    report_lines.append("")
    report_lines.append("Core leakage checks:")
    report_lines.append(f"- overlapping example_id between train and test: {len(overlap_example_ids)}")
    report_lines.append(f"- overlapping fact_id between train and test: {len(overlap_fact_ids)}")
    report_lines.append(f"- exact duplicate input texts between train and test: {len(overlap_inputs)}")
    report_lines.append(f"- exact duplicate utterances between train and test: {len(overlap_utterances)}")
    report_lines.append(f"- exact duplicate base_fact texts between train and test: {len(overlap_base_facts)}")
    report_lines.append("")
    report_lines.append("User-level overlap check:")
    report_lines.append(f"- overlapping user_key/dialogue_id+speaker between train and test: {len(overlap_users)}")
    report_lines.append("")
    report_lines.append("Interpretation:")
    if len(overlap_fact_ids) == 0 and len(overlap_inputs) == 0:
        report_lines.append(
            "- No direct leakage was found at the fact_id level or exact input-text level."
        )
    else:
        report_lines.append(
            "- Potential direct leakage was found. Inspect the overlap CSV files."
        )

    if len(overlap_base_facts) > 0:
        report_lines.append(
            "- Some base_fact texts appear in both train and test. This may happen if the same fact text occurs for different users, but it should be reported."
        )

    if len(overlap_utterances) > 0:
        report_lines.append(
            "- Some utterances appear in both train and test. Inspect overlap_utterances.csv."
        )

    if len(overlap_users) > 0:
        report_lines.append(
            "- The current split is grouped by fact_id, not by user. Therefore, some users/dialogues may appear in both train and test. This is not direct leakage for the pairwise fact-level task, but a stricter user-level split could be used as an additional robustness check."
        )
    else:
        report_lines.append(
            "- No user-level overlap was found."
        )

    report = "\n".join(report_lines)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()