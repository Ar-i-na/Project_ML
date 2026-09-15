import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

def compute_metrics(y_true, y_pred, target_names):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    return {
        "macro_f1": macro_f1,
        "accuracy": report["accuracy"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in target_names
        },
    }


def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(f"\n=== {title} ===")
    print(f"macro-F1: {metrics['macro_f1']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print("По классам:")
    for name, vals in metrics["per_class"].items():
        print(
            f"  {name:25s} precision={vals['precision']:.3f} "
            f"recall={vals['recall']:.3f} f1={vals['f1']:.3f} "
            f"(n={int(vals['support'])})"
        )


def plot_confusion_matrix(y_true, y_pred, target_names, save_path=None, title=""):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names, ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title or "Confusion matrix (нормализована по строкам)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix сохранена: {save_path}")
    plt.close(fig)
    return cm
