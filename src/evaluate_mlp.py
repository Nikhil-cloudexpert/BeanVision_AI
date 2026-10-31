from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = ROOT / "data" / "processed" / "test_data.npz"
MODEL_FILE = ROOT / "models" / "MLP" / "mlp_classifier.pth"

OUTPUT_DIR = ROOT / "outputs"
CM_DIR = OUTPUT_DIR / "confusion_matrix"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOGS_DIR = OUTPUT_DIR / "logs"

for folder in (CM_DIR, METRICS_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "cut",
    "good",
    "husk",
    "immature",
    "parchment",
    "partial-black",
    "shell",
]

NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Model architecture
# Must match the architecture used during training.
# --------------------------------------------------

class CoffeeMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(768, 512),   # model.0
            nn.BatchNorm1d(512),   # model.1
            nn.ReLU(),             # model.2
            nn.Dropout(0.3),       # model.3
            nn.Linear(512, 256),   # model.4
            nn.BatchNorm1d(256),   # model.5
            nn.ReLU(),             # model.6
            nn.Dropout(0.3),       # model.7
            nn.Linear(256, 7),     # model.8
        )

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Load test data
# --------------------------------------------------

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test data not found: {TEST_FILE}")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_FILE}")

data = np.load(TEST_FILE)

X = data["features"].astype(np.float32)
y_true = data["labels"].astype(np.int64)

if X.ndim != 2 or X.shape[1] != 768:
    raise ValueError(f"Expected features shaped (N, 768), got {X.shape}")

if len(X) != len(y_true):
    raise ValueError("Number of feature vectors and labels does not match.")

if len(y_true) == 0:
    raise ValueError("Test dataset is empty.")

if np.any((y_true < 0) | (y_true >= NUM_CLASSES)):
    raise ValueError("Test labels contain class IDs outside 0–6.")

print(f"Test samples: {len(y_true)}")
print(f"Feature shape: {X.shape}")
print(f"Device: {DEVICE}")


# --------------------------------------------------
# Load checkpoint and predict
# --------------------------------------------------

model = CoffeeMLP().to(DEVICE)

state_dict = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

X_tensor = torch.from_numpy(X).to(DEVICE)

with torch.inference_mode():
    logits = model(X_tensor)
    probabilities = torch.softmax(logits, dim=1)
    y_pred = torch.argmax(probabilities, dim=1).cpu().numpy()
    confidence = probabilities.max(dim=1).values.cpu().numpy()


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    zero_division=0,
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
)

summary = {
    "evaluation": "MLP classifier on saved DINOv2 test features",
    "test_samples": int(len(y_true)),
    "feature_dimensions": int(X.shape[1]),
    "accuracy": float(accuracy),
    "macro_precision": float(np.mean(precision)),
    "macro_recall": float(np.mean(recall)),
    "macro_f1": float(np.mean(f1)),
    "weighted_f1": float(
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )[2]
    ),
    "device": str(DEVICE),
    "class_names": CLASS_NAMES,
}

# JSON summary
with open(METRICS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# Per-class report
import csv

with open(
    METRICS_DIR / "classification_report.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["class", "precision", "recall", "f1_score", "support"],
    )
    writer.writeheader()

    for name in CLASS_NAMES:
        row = report[name]
        writer.writerow({
            "class": name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1-score"],
            "support": row["support"],
        })

    for name in ("macro avg", "weighted avg"):
        row = report[name]
        writer.writerow({
            "class": name,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1-score"],
            "support": row["support"],
        })

# Confusion matrix values
np.savetxt(
    CM_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)

# Regular confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)
display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title("MLP Test Confusion Matrix")
fig.tight_layout()
fig.savefig(CM_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Normalized confusion matrix
row_sums = cm.sum(axis=1, keepdims=True)
normalized_cm = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0,
)

fig, ax = plt.subplots(figsize=(10, 8))
display = ConfusionMatrixDisplay(
    confusion_matrix=normalized_cm,
    display_labels=CLASS_NAMES,
)
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format=".2f",
    colorbar=False,
)
ax.set_title("Normalized MLP Test Confusion Matrix")
fig.tight_layout()
fig.savefig(
    CM_DIR / "confusion_matrix_normalized.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# Predictions for each test instance
with open(
    OUTPUT_DIR / "predictions.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "test_index",
        "true_label_id",
        "true_class",
        "predicted_label_id",
        "predicted_class",
        "confidence",
        "correct",
    ])

    for i, (true_id, pred_id, conf) in enumerate(
        zip(y_true, y_pred, confidence)
    ):
        writer.writerow([
            i,
            int(true_id),
            CLASS_NAMES[true_id],
            int(pred_id),
            CLASS_NAMES[pred_id],
            float(conf),
            bool(true_id == pred_id),
        ])

# Text summary
summary_lines = [
    "BeanVision AI - MLP Test Evaluation",
    "=" * 42,
    f"Test samples: {len(y_true)}",
    f"Accuracy: {accuracy:.4f}",
    f"Macro precision: {summary['macro_precision']:.4f}",
    f"Macro recall: {summary['macro_recall']:.4f}",
    f"Macro F1: {summary['macro_f1']:.4f}",
    f"Weighted F1: {summary['weighted_f1']:.4f}",
    "",
    "Classification report:",
    classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ),
]

(LOGS_DIR / "mlp_evaluation.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\n=== MLP Test Results ===")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0,
))
print(f"Accuracy: {accuracy:.4f}")
print(f"\nResults saved under: {OUTPUT_DIR}")