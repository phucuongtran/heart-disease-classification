from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heart_disease.config import OUTPUT_DIR, PROCESSED_DIR, SEED
from heart_disease.data_loader import load_processed_splits
from heart_disease.modeling import build_stacking, fit_predict, run_all_models
from heart_disease.visualization import save_bar_chart, save_confusion_matrix

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    datasets = load_processed_splits(PROCESSED_DIR)
    results_df = run_all_models(datasets)
    results_path = OUTPUT_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    best = results_df.iloc[0]
    required_names = {"GaussianNB", "KNN", "DecisionTree", "KMeans-2", "Stacking"}
    required_df = results_df[results_df["model"].isin(required_names)].copy()
    required_best = required_df.iloc[0]

    print("=== BEST OVERALL MODEL ===")
    print(best[["model", "dataset", "val_accuracy", "test_accuracy", "val_f1", "test_f1", "details"]])
    print()

    print("=== BEST REQUIRED-COURSE MODEL ===")
    print(required_best[["model", "dataset", "val_accuracy", "test_accuracy", "val_f1", "test_f1", "details"]])
    print()

    save_bar_chart(required_df, OUTPUT_DIR / "required_models_test_accuracy.png", "Required Project Models - Test Accuracy")
    save_bar_chart(results_df, OUTPUT_DIR / "all_models_test_accuracy.png", "All Evaluated Models - Test Accuracy")

    best_ds = datasets[best["dataset"]]
    if best["model"] == "GaussianNB":
        _, _, test_pred = fit_predict(GaussianNB(), best_ds)
    elif best["model"] == "KNN":
        best_k = int(str(best["details"]).split("best_k=")[1].split(",")[0])
        _, _, test_pred = fit_predict(KNeighborsClassifier(n_neighbors=best_k), best_ds)
    elif best["model"] == "DecisionTree":
        best_depth = int(str(best["details"]).split("best_depth=")[1].split(",")[0])
        _, _, test_pred = fit_predict(DecisionTreeClassifier(max_depth=best_depth, random_state=SEED), best_ds)
    elif best["model"] == "Stacking":
        _, _, test_pred = fit_predict(build_stacking(), best_ds)
    elif best["model"] == "LogReg":
        _, _, test_pred = fit_predict(LogisticRegression(max_iter=4000, random_state=SEED), best_ds)
    elif best["model"] == "RandomForest":
        _, _, test_pred = fit_predict(RandomForestClassifier(n_estimators=300, random_state=SEED), best_ds)
    elif best["model"] == "ExtraTrees":
        _, _, test_pred = fit_predict(ExtraTreesClassifier(n_estimators=600, random_state=SEED), best_ds)
    elif best["model"] == "SVC_rbf":
        _, _, test_pred = fit_predict(SVC(C=1.0, kernel="rbf", random_state=SEED), best_ds)
    elif best["model"] == "SVC_linear":
        _, _, test_pred = fit_predict(SVC(C=1.0, kernel="linear", random_state=SEED), best_ds)
    else:
        raise ValueError(f"Unsupported model for confusion matrix: {best['model']}")

    save_confusion_matrix(
        best_ds.y_test,
        test_pred,
        title=f"Confusion Matrix - {best['model']} ({best['dataset']})",
        output_path=OUTPUT_DIR / "best_model_confusion_matrix.png",
    )

    summary = {
        "best_overall": best.to_dict(),
        "best_required": required_best.to_dict(),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
