import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_PATH = Path("01_dataset_final.jsonl")

OUTPUT_DIR = Path("data/diversity_embedding")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "embedding_diversity_report.txt"
SIMILAR_PAIRS_CSV = OUTPUT_DIR / "embedding_similar_pairs.csv"
NEAREST_NEIGHBORS_CSV = OUTPUT_DIR / "embedding_nearest_neighbors.csv"
PCA_PLOT_PATH = OUTPUT_DIR / "embedding_pca_plot.png"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

BATCH_SIZE = 32

# cosine similarity thresholds
HIGH_SIM_THRESHOLD = 0.85
VERY_HIGH_SIM_THRESHOLD = 0.90

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
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_operation(row):
    return row.get("target_operation") or row.get("gold_operation")


def get_utterance(row):
    return str(row.get("utterance", "")).strip()


def encode_texts(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


def compute_nearest_neighbors(df, embeddings):
    sim = cosine_similarity(embeddings, embeddings)

    # self-similarity убираем
    np.fill_diagonal(sim, -1.0)

    nearest_idx = sim.argmax(axis=1)
    nearest_score = sim.max(axis=1)

    rows = []

    for i, j in enumerate(nearest_idx):
        rows.append({
            "example_id": df.loc[i, "example_id"],
            "operation": df.loc[i, "operation"],
            "utterance": df.loc[i, "utterance"],
            "nearest_example_id": df.loc[j, "example_id"],
            "nearest_operation": df.loc[j, "operation"],
            "nearest_utterance": df.loc[j, "utterance"],
            "cosine_similarity": float(nearest_score[i]),
            "same_operation": df.loc[i, "operation"] == df.loc[j, "operation"],
        })

    return pd.DataFrame(rows), sim


def collect_similar_pairs(df, sim):
    pairs = []

    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):
            score = sim[i, j]

            if score >= HIGH_SIM_THRESHOLD:
                pairs.append({
                    "cosine_similarity": float(score),
                    "example_id_1": df.loc[i, "example_id"],
                    "operation_1": df.loc[i, "operation"],
                    "utterance_1": df.loc[i, "utterance"],
                    "example_id_2": df.loc[j, "example_id"],
                    "operation_2": df.loc[j, "operation"],
                    "utterance_2": df.loc[j, "utterance"],
                    "same_operation": df.loc[i, "operation"] == df.loc[j, "operation"],
                })

    pairs_df = pd.DataFrame(pairs)

    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("cosine_similarity", ascending=False)

    return pairs_df


def make_pca_plot(df, embeddings):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    plot_df = df.copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))

    for label in LABELS:
        sub = plot_df[plot_df["operation"] == label]

        if len(sub) == 0:
            continue

        ax.scatter(
            sub["pc1"],
            sub["pc2"],
            label=label,
            alpha=0.65,
            s=24,
        )

    ax.set_title("PCA of utterance embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    return pca.explained_variance_ratio_


def main():
    rows = load_jsonl(INPUT_PATH)

    data = []

    for row in rows:
        utterance = get_utterance(row)
        operation = get_operation(row)

        if not utterance or operation not in LABELS:
            continue

        data.append({
            "example_id": row.get("example_id"),
            "fact_id": row.get("fact_id"),
            "operation": operation,
            "utterance": utterance,
            "base_fact": row.get("base_fact"),
        })

    df = pd.DataFrame(data)

    print(f"Loaded examples: {len(df)}")
    print(df["operation"].value_counts().sort_index())

    embeddings = encode_texts(df["utterance"].tolist())

    nearest_df, sim = compute_nearest_neighbors(df, embeddings)
    nearest_df.to_csv(NEAREST_NEIGHBORS_CSV, index=False, encoding="utf-8-sig")

    similar_pairs_df = collect_similar_pairs(df, sim)
    similar_pairs_df.to_csv(SIMILAR_PAIRS_CSV, index=False, encoding="utf-8-sig")

    pca_var = make_pca_plot(df, embeddings)

    report_lines = []

    report_lines.append("EMBEDDING DIVERSITY REPORT")
    report_lines.append("==========================")
    report_lines.append("")
    report_lines.append(f"Input file: {INPUT_PATH}")
    report_lines.append(f"Embedding model: {MODEL_NAME}")
    report_lines.append(f"Total examples: {len(df)}")
    report_lines.append("")
    report_lines.append("Class distribution:")
    report_lines.append(df["operation"].value_counts().sort_index().to_string())
    report_lines.append("")

    report_lines.append("Nearest-neighbor cosine similarity:")
    report_lines.append(f"- mean: {nearest_df['cosine_similarity'].mean():.4f}")
    report_lines.append(f"- median: {nearest_df['cosine_similarity'].median():.4f}")
    report_lines.append(f"- std: {nearest_df['cosine_similarity'].std():.4f}")
    report_lines.append(f"- min: {nearest_df['cosine_similarity'].min():.4f}")
    report_lines.append(f"- max: {nearest_df['cosine_similarity'].max():.4f}")
    report_lines.append("")

    high_count = int((nearest_df["cosine_similarity"] >= HIGH_SIM_THRESHOLD).sum())
    very_high_count = int((nearest_df["cosine_similarity"] >= VERY_HIGH_SIM_THRESHOLD).sum())

    report_lines.append("High-similarity nearest neighbors:")
    report_lines.append(f"- examples with NN cosine >= {HIGH_SIM_THRESHOLD}: {high_count}")
    report_lines.append(f"- examples with NN cosine >= {VERY_HIGH_SIM_THRESHOLD}: {very_high_count}")
    report_lines.append("")

    report_lines.append("Similar pairs:")
    report_lines.append(f"- pairs with cosine >= {HIGH_SIM_THRESHOLD}: {len(similar_pairs_df)}")

    if len(similar_pairs_df) > 0:
        same_op = int(similar_pairs_df["same_operation"].sum())
        cross_op = len(similar_pairs_df) - same_op

        report_lines.append(f"- same-operation similar pairs: {same_op}")
        report_lines.append(f"- cross-operation similar pairs: {cross_op}")

    report_lines.append("")

    report_lines.append("Average nearest-neighbor similarity by operation:")
    by_op = nearest_df.groupby("operation")["cosine_similarity"].agg(
        ["mean", "median", "max", "count"]
    ).sort_index()
    report_lines.append(by_op.to_string())
    report_lines.append("")

    report_lines.append("Examples whose nearest neighbor is from another operation:")
    cross_nn = nearest_df[nearest_df["same_operation"] == False]
    report_lines.append(f"- count: {len(cross_nn)}")
    report_lines.append(f"- share: {len(cross_nn) / len(nearest_df):.4f}")
    report_lines.append("")

    if len(cross_nn) > 0:
        report_lines.append("Top 20 cross-operation nearest neighbors:")
        top_cross = cross_nn.sort_values("cosine_similarity", ascending=False).head(20)

        for _, row in top_cross.iterrows():
            report_lines.append("")
            report_lines.append(f"Cosine: {row['cosine_similarity']:.4f}")
            report_lines.append(f"1 [{row['operation']}]: {row['utterance']}")
            report_lines.append(f"2 [{row['nearest_operation']}]: {row['nearest_utterance']}")

    report_lines.append("")
    report_lines.append("PCA:")
    report_lines.append(f"- PC1 explained variance: {pca_var[0]:.4f}")
    report_lines.append(f"- PC2 explained variance: {pca_var[1]:.4f}")
    report_lines.append(f"- PCA plot saved to: {PCA_PLOT_PATH}")
    report_lines.append("")
    report_lines.append("Saved files:")
    report_lines.append(f"- nearest neighbors: {NEAREST_NEIGHBORS_CSV}")
    report_lines.append(f"- similar pairs: {SIMILAR_PAIRS_CSV}")
    report_lines.append(f"- PCA plot: {PCA_PLOT_PATH}")

    report_text = "\n".join(report_lines)

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print()
    print(report_text)
    print()
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()