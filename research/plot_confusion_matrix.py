from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# путь к confusion matrix
INPUT_PATH = Path("data/roberta_multifact_classifier/roberta_multifact_from_final_confusion_matrix.csv")
OUTPUT_DIR = Path("data/roberta_multifact_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# читаем матрицу
df = pd.read_csv(INPUT_PATH, index_col=0)

# убираем префиксы true_ / pred_ для красоты
df.index = [x.replace("true_", "") for x in df.index]
df.columns = [x.replace("pred_", "") for x in df.columns]

# ---------- обычная heatmap ----------
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(df.values)

ax.set_xticks(range(len(df.columns)))
ax.set_yticks(range(len(df.index)))
ax.set_xticklabels(df.columns, rotation=45, ha="right")
ax.set_yticklabels(df.index)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix")

# подписи в ячейках
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        ax.text(j, i, str(df.iloc[i, j]), ha="center", va="center")

fig.colorbar(im, ax=ax)
plt.tight_layout()

out_path_1 = OUTPUT_DIR / "confusion_matrix_heatmap.png"
plt.savefig(out_path_1, dpi=200, bbox_inches="tight")
plt.savefig(out_path_1, dpi=200, bbox_inches="tight")
plt.close()

# ---------- нормализованная heatmap ----------
df_norm = df.div(df.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(df_norm.values)

ax.set_xticks(range(len(df_norm.columns)))
ax.set_yticks(range(len(df_norm.index)))
ax.set_xticklabels(df_norm.columns, rotation=45, ha="right")
ax.set_yticklabels(df_norm.index)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Normalized Confusion Matrix")

# подписи в ячейках
for i in range(df_norm.shape[0]):
    for j in range(df_norm.shape[1]):
        ax.text(j, i, f"{df_norm.iloc[i, j]:.2f}", ha="center", va="center")

fig.colorbar(im, ax=ax)
plt.tight_layout()

out_path_2 = OUTPUT_DIR / "confusion_matrix_normalized_heatmap.png"
plt.savefig(out_path_2, dpi=200, bbox_inches="tight")
plt.savefig(out_path_2, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:")
print(out_path_1)
print(out_path_2)