from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import RAW_COLUMNS


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def read_split_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return X, y


def load_processed_splits(base_dir: Path) -> Dict[str, DatasetBundle]:
    raw_train_X, raw_train_y = read_split_csv(base_dir / "raw_train.csv")
    raw_val_X, raw_val_y = read_split_csv(base_dir / "raw_val.csv")
    raw_test_X, raw_test_y = read_split_csv(base_dir / "raw_test.csv")

    fe_train_X, fe_train_y = read_split_csv(base_dir / "fe_train.csv")
    fe_val_X, fe_val_y = read_split_csv(base_dir / "fe_val.csv")
    fe_test_X, fe_test_y = read_split_csv(base_dir / "fe_test.csv")

    return {
        "raw": DatasetBundle(raw_train_X, raw_train_y, raw_val_X, raw_val_y, raw_test_X, raw_test_y),
        "fe": DatasetBundle(fe_train_X, fe_train_y, fe_val_X, fe_val_y, fe_test_X, fe_test_y),
    }


def read_original_cleveland(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    df.columns = RAW_COLUMNS
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)
    return df
