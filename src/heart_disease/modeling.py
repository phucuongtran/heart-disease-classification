import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import SEED
from .data_loader import DatasetBundle


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def fit_predict(model, ds: DatasetBundle):
    trained = clone(model)
    trained.fit(ds.X_train, ds.y_train)
    val_pred = trained.predict(ds.X_val)
    test_pred = trained.predict(ds.X_test)
    return trained, val_pred, test_pred


def tune_knn(X: pd.DataFrame, y: pd.Series, k_max: int = 30) -> Tuple[int, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_k, best_score = None, -1.0
    for k in range(1, k_max + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        scores = []
        for tr_idx, va_idx in cv.split(X, y):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            scores.append(accuracy_score(y.iloc[va_idx], pred))
        score = float(np.mean(scores))
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def tune_dt(X: pd.DataFrame, y: pd.Series, max_depth_max: int = 15) -> Tuple[int, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_depth, best_score = None, -1.0
    for depth in range(1, max_depth_max + 1):
        model = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        scores = []
        for tr_idx, va_idx in cv.split(X, y):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[va_idx])
            scores.append(accuracy_score(y.iloc[va_idx], pred))
        score = float(np.mean(scores))
        if score > best_score:
            best_depth, best_score = depth, score
    return best_depth, best_score


def build_stacking() -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier()),
            ("dt", DecisionTreeClassifier(random_state=SEED)),
            ("nb", GaussianNB()),
        ],
        final_estimator=KNeighborsClassifier(),
        stack_method="predict_proba",
        passthrough=False,
    )


def evaluate_kmeans(ds: DatasetBundle, dataset_name: str):
    model = KMeans(n_clusters=2, random_state=SEED, n_init=20)
    model.fit(ds.X_train)

    train_clusters = model.labels_
    mapping = {}
    for cluster_id in np.unique(train_clusters):
        idx = train_clusters == cluster_id
        mapping[int(cluster_id)] = int(pd.Series(ds.y_train[idx]).value_counts().idxmax())

    val_clusters = model.predict(ds.X_val)
    test_clusters = model.predict(ds.X_test)
    val_pred = np.array([mapping[int(c)] for c in val_clusters])
    test_pred = np.array([mapping[int(c)] for c in test_clusters])

    return {
        "model": "KMeans-2",
        "dataset": dataset_name,
        **{f"val_{k}": v for k, v in calc_metrics(ds.y_val, val_pred).items()},
        **{f"test_{k}": v for k, v in calc_metrics(ds.y_test, test_pred).items()},
        "details": json.dumps(mapping, ensure_ascii=False),
    }


def run_all_models(datasets: Dict[str, DatasetBundle]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for dataset_name, ds in datasets.items():
        _, val_pred, test_pred = fit_predict(GaussianNB(), ds)
        rows.append({
            "model": "GaussianNB",
            "dataset": dataset_name,
            **{f"val_{k}": v for k, v in calc_metrics(ds.y_val, val_pred).items()},
            **{f"test_{k}": v for k, v in calc_metrics(ds.y_test, test_pred).items()},
            "details": "course baseline",
        })

        best_k, cv_acc = tune_knn(ds.X_train, ds.y_train)
        _, val_pred, test_pred = fit_predict(KNeighborsClassifier(n_neighbors=best_k), ds)
        rows.append({
            "model": "KNN",
            "dataset": dataset_name,
            **{f"val_{k}": v for k, v in calc_metrics(ds.y_val, val_pred).items()},
            **{f"test_{k}": v for k, v in calc_metrics(ds.y_test, test_pred).items()},
            "details": f"best_k={best_k}, cv_acc={cv_acc:.4f}",
        })

        best_depth, cv_acc = tune_dt(ds.X_train, ds.y_train)
        _, val_pred, test_pred = fit_predict(DecisionTreeClassifier(max_depth=best_depth, random_state=SEED), ds)
        rows.append({
            "model": "DecisionTree",
            "dataset": dataset_name,
            **{f"val_{k}": v for k, v in calc_metrics(ds.y_val, val_pred).items()},
            **{f"test_{k}": v for k, v in calc_metrics(ds.y_test, test_pred).items()},
            "details": f"best_depth={best_depth}, cv_acc={cv_acc:.4f}",
        })

        rows.append(evaluate_kmeans(ds, dataset_name))

        _, val_pred, test_pred = fit_predict(build_stacking(), ds)
        rows.append({
            "model": "Stacking",
            "dataset": dataset_name,
            **{f"val_{k}": v for k, v in calc_metrics(ds.y_val, val_pred).items()},
            **{f"test_{k}": v for k, v in calc_metrics(ds.y_test, test_pred).items()},
            "details": "base=[KNN,DT,NB], meta=KNN",
        })

        bonus_models = {
            "LogReg": LogisticRegression(max_iter=4000, random_state=SEED),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=600, random_state=SEED),
            "SVC_rbf": SVC(C=1.0, kernel="rbf", random_state=SEED),
            "SVC_linear": SVC(C=1.0, kernel="linear", random_state=SEED),
        }
        for model_name, base_model in bonus_models.items():
            _, val_pred, test_pred = fit_predict(base_model, ds)
            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                **{f"val_{k}": v for k, v in calc_metrics(ds.y_val, val_pred).items()},
                **{f"test_{k}": v for k, v in calc_metrics(ds.y_test, test_pred).items()},
                "details": "bonus optimization",
            })

    return pd.DataFrame(rows).sort_values(["test_accuracy", "val_accuracy", "test_f1"], ascending=False).reset_index(drop=True)
