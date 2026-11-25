import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


INPUT_PATH = Path("01_dataset_final.jsonl")

OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUTPUT_DIR / "diversity_summary.txt"
SIMILAR_PAIRS_PATH = OUTPUT_DIR / "similar_pairs.csv"
FEATURES_PATH = OUTPUT_DIR / "utterance_features.csv"

# Порог похожести.
# Чем выше, тем строже.
# 0.50 значит: показываем пары, у которых пересечение лемм довольно большое.
MIN_JACCARD = 0.50

# Сколько самых частых элементов показывать в summary.
TOP_N = 30


STOPWORDS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers",
    "it", "its", "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "a", "an", "the",
    "and", "or", "but", "if", "because", "so",
    "to", "of", "in", "on", "at", "for", "from", "with", "about",
    "as", "by", "into", "over", "after", "before", "during",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "can", "could", "should", "would", "will", "may", "might", "must",
    "not", "no",
    "this", "that", "these", "those",
    "there", "here",
    "just", "really", "very", "quite", "also", "actually",
    "now", "recently", "usually", "often", "sometimes",
}


def simple_lemma(token: str) -> str:
    """
    Очень простой fallback-лемматизатор без внешних библиотек.

    Это не идеальная настоящая лемматизация, но для диагностики повторов работает:
    - lower-case
    - убирает частые английские окончания
    """
    token = token.lower().strip()

    irregular = {
        "went": "go",
        "gone": "go",
        "better": "good",
        "best": "good",
        "worse": "bad",
        "worst": "bad",
        "children": "child",
        "people": "person",
        "men": "man",
        "women": "woman",
        "mice": "mouse",
        "feet": "foot",
        "teeth": "tooth",
    }

    if token in irregular:
        return irregular[token]

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("es"):
        return token[:-2]

    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token


def tokenize(text: str) -> list[str]:
    """
    Достаёт слова из текста.
    Например:
    "I've started painting." -> ["i", "ve", "started", "painting"]
    """
    return re.findall(r"[A-Za-z']+", text.lower())


def get_content_lemmas(text: str) -> list[str]:
    tokens = tokenize(text)

    lemmas = []

    for token in tokens:
        token = token.strip("'")

        if not token:
            continue

        lemma = simple_lemma(token)

        if lemma in STOPWORDS:
            continue

        if len(lemma) <= 2:
            continue

        lemmas.append(lemma)

    return lemmas


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def get_first_words(text: str, n: int) -> str:
    tokens = tokenize(text)
    return " ".join(tokens[:n])


def load_generated_examples(path: Path) -> list[dict]:
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)

            utterance = item.get("utterance")

            # На случай, если utterance лежит только внутри parsed_response.
            if not utterance:
                parsed = item.get("parsed_response")
                if isinstance(parsed, dict):
                    utterance = parsed.get("utterance")

            if not utterance:
                continue

            examples.append({
                "line_number": line_number,
                "example_id": item.get("example_id", ""),
                "fact_id": item.get("fact_id", ""),
                "base_fact": item.get("base_fact", ""),
                "target_operation": item.get("target_operation", ""),
                "tags": item.get("tags", []),
                "utterance": utterance.strip(),
                "parse_status": item.get("parse_status", ""),
            })

    return examples


def write_features(examples: list[dict]):
    with open(FEATURES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "line_number",
                "example_id",
                "fact_id",
                "target_operation",
                "tags",
                "utterance",
                "first_1",
                "first_2",
                "first_3",
                "lemmas",
                "unique_lemmas_count",
            ]
        )

        writer.writeheader()

        for ex in examples:
            writer.writerow({
                "line_number": ex["line_number"],
                "example_id": ex["example_id"],
                "fact_id": ex["fact_id"],
                "target_operation": ex["target_operation"],
                "tags": "|".join(ex["tags"]) if isinstance(ex["tags"], list) else ex["tags"],
                "utterance": ex["utterance"],
                "first_1": get_first_words(ex["utterance"], 1),
                "first_2": get_first_words(ex["utterance"], 2),
                "first_3": get_first_words(ex["utterance"], 3),
                "lemmas": " ".join(ex["lemmas"]),
                "unique_lemmas_count": len(set(ex["lemmas"])),
            })


def write_similar_pairs(examples: list[dict]):
    similar_pairs = []

    for ex1, ex2 in combinations(examples, 2):
        set1 = set(ex1["lemmas"])
        set2 = set(ex2["lemmas"])

        score = jaccard(set1, set2)

        if score >= MIN_JACCARD:
            shared_lemmas = sorted(set1 & set2)

            similar_pairs.append({
                "jaccard": round(score, 4),
                "shared_lemmas": " ".join(shared_lemmas),

                "example_id_1": ex1["example_id"],
                "fact_id_1": ex1["fact_id"],
                "target_operation_1": ex1["target_operation"],
                "utterance_1": ex1["utterance"],

                "example_id_2": ex2["example_id"],
                "fact_id_2": ex2["fact_id"],
                "target_operation_2": ex2["target_operation"],
                "utterance_2": ex2["utterance"],
            })

    similar_pairs.sort(key=lambda x: x["jaccard"], reverse=True)

    with open(SIMILAR_PAIRS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "jaccard",
                "shared_lemmas",
                "example_id_1",
                "fact_id_1",
                "target_operation_1",
                "utterance_1",
                "example_id_2",
                "fact_id_2",
                "target_operation_2",
                "utterance_2",
            ]
        )

        writer.writeheader()
        writer.writerows(similar_pairs)

    return similar_pairs


def calculate_average_max_similarity(examples: list[dict]) -> dict:
    """
    Для каждой операции считает:
    насколько в среднем каждый пример похож на самый похожий другой пример
    внутри той же операции.

    Если значение высокое, значит внутри операции много однотипных реплик.
    """
    by_operation = defaultdict(list)

    for ex in examples:
        by_operation[ex["target_operation"]].append(ex)

    result = {}

    for operation, operation_examples in by_operation.items():
        max_scores = []

        for ex in operation_examples:
            scores = []

            for other in operation_examples:
                if ex["example_id"] == other["example_id"]:
                    continue

                score = jaccard(set(ex["lemmas"]), set(other["lemmas"]))
                scores.append(score)

            if scores:
                max_scores.append(max(scores))

        if max_scores:
            result[operation] = sum(max_scores) / len(max_scores)
        else:
            result[operation] = 0.0

    return result


def write_summary(examples: list[dict], similar_pairs: list[dict]):
    utterances = [ex["utterance"] for ex in examples]

    first_1_counter = Counter(get_first_words(u, 1) for u in utterances if get_first_words(u, 1))
    first_2_counter = Counter(get_first_words(u, 2) for u in utterances if get_first_words(u, 2))
    first_3_counter = Counter(get_first_words(u, 3) for u in utterances if get_first_words(u, 3))

    lemma_counter = Counter()

    operation_counter = Counter()

    exact_duplicate_counter = Counter(utterances)

    for ex in examples:
        lemma_counter.update(ex["lemmas"])
        operation_counter[ex["target_operation"]] += 1

    duplicated_utterances = {
        utterance: count
        for utterance, count in exact_duplicate_counter.items()
        if count > 1
    }

    avg_max_similarity = calculate_average_max_similarity(examples)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("DIVERSITY REPORT\n")
        f.write("================\n\n")

        f.write(f"Input file: {INPUT_PATH}\n")
        f.write(f"Total examples: {len(examples)}\n")
        f.write(f"Similar pairs threshold: {MIN_JACCARD}\n")
        f.write(f"Similar pairs found: {len(similar_pairs)}\n\n")

        f.write("Examples by target_operation:\n")
        for operation, count in operation_counter.most_common():
            f.write(f"- {operation}: {count}\n")
        f.write("\n")

        f.write("Exact duplicate utterances:\n")
        if duplicated_utterances:
            for utterance, count in duplicated_utterances.items():
                f.write(f"- {count}x: {utterance}\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write("Average max Jaccard similarity within each operation:\n")
        f.write("Higher values mean more repetitive examples inside the same operation.\n")
        for operation, value in sorted(avg_max_similarity.items()):
            f.write(f"- {operation}: {value:.4f}\n")
        f.write("\n")

        f.write(f"Top {TOP_N} first 1-word openings:\n")
        for phrase, count in first_1_counter.most_common(TOP_N):
            f.write(f"- {count}: {phrase}\n")
        f.write("\n")

        f.write(f"Top {TOP_N} first 2-word openings:\n")
        for phrase, count in first_2_counter.most_common(TOP_N):
            f.write(f"- {count}: {phrase}\n")
        f.write("\n")

        f.write(f"Top {TOP_N} first 3-word openings:\n")
        for phrase, count in first_3_counter.most_common(TOP_N):
            f.write(f"- {count}: {phrase}\n")
        f.write("\n")

        f.write(f"Top {TOP_N} content lemmas:\n")
        for lemma, count in lemma_counter.most_common(TOP_N):
            f.write(f"- {count}: {lemma}\n")
        f.write("\n")

        f.write("Most similar pairs preview:\n")
        for pair in similar_pairs[:TOP_N]:
            f.write("\n")
            f.write(f"Jaccard: {pair['jaccard']}\n")
            f.write(f"Shared lemmas: {pair['shared_lemmas']}\n")
            f.write(f"1 [{pair['target_operation_1']}]: {pair['utterance_1']}\n")
            f.write(f"2 [{pair['target_operation_2']}]: {pair['utterance_2']}\n")


def main():
    examples = load_generated_examples(INPUT_PATH)

    if not examples:
        print(f"No examples found in {INPUT_PATH}")
        return

    for ex in examples:
        ex["lemmas"] = get_content_lemmas(ex["utterance"])

    similar_pairs = write_similar_pairs(examples)
    write_features(examples)
    write_summary(examples, similar_pairs)

    print("Done.")
    print(f"Loaded examples: {len(examples)}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Similar pairs: {SIMILAR_PAIRS_PATH}")
    print(f"Features: {FEATURES_PATH}")


if __name__ == "__main__":
    main()