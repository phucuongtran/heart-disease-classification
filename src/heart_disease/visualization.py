from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bar_chart(df, output_path: Path, title: str) -> None:
    labels = df["model"] + " (" + df["dataset"] + ")"
    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))
    plt.bar(x, df["test_accuracy"])
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
