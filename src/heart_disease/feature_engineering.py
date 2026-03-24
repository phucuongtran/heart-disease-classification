import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLS, NUMERIC_COLS, SEED
from .data_loader import DatasetBundle, read_original_cleveland


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"chol", "age"} <= set(df.columns):
        df["chol_per_age"] = df["chol"] / df["age"]
    if {"trestbps", "age"} <= set(df.columns):
        df["bps_per_age"] = df["trestbps"] / df["age"]
    if {"thalach", "age"} <= set(df.columns):
        df["hr_ratio"] = df["thalach"] / df["age"]
    if "age" in df.columns:
        df["age_bin"] = pd.cut(df["age"], bins=5, labels=False).astype("category")
    return df


def rebuild_splits_from_original(cleveland_csv):
    raw = read_original_cleveland(cleveland_csv)
    y_all = raw["target"]
    X_all = raw.drop(columns=["target"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    raw_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    raw_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", MinMaxScaler()),
    ])
    raw_pre = ColumnTransformer([
        ("num", raw_num, NUMERIC_COLS),
        ("cat", raw_cat, CATEGORICAL_COLS),
    ], verbose_feature_names_out=False).set_output(transform="pandas")

    X_raw_train = raw_pre.fit_transform(X_train, y_train)
    X_raw_val = raw_pre.transform(X_val)
    X_raw_test = raw_pre.transform(X_test)

    gen_num = ["chol_per_age", "bps_per_age", "hr_ratio"]
    gen_cat = ["age_bin"]
    all_num = NUMERIC_COLS + gen_num
    all_cat = CATEGORICAL_COLS + gen_cat

    fe_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    fe_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    fe_pre = ColumnTransformer([
        ("num", fe_num, all_num),
        ("cat", fe_cat, all_cat),
    ], verbose_feature_names_out=False).set_output(transform="pandas")

    fe_pipeline = Pipeline([
        ("add_features", FunctionTransformer(add_new_features, validate=False)),
        ("preprocess", fe_pre),
    ]).set_output(transform="pandas")

    Xt_tr = fe_pipeline.fit_transform(X_train, y_train)
    Xt_va = fe_pipeline.transform(X_val)
    Xt_te = fe_pipeline.transform(X_test)

    non_constant_cols = Xt_tr.columns[Xt_tr.nunique(dropna=False) > 1]
    Xt_tr, Xt_va, Xt_te = Xt_tr[non_constant_cols], Xt_va[non_constant_cols], Xt_te[non_constant_cols]

    ohe = fe_pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out(all_cat))
    is_discrete = np.array([col in cat_feature_names for col in Xt_tr.columns], dtype=bool)

    mi = mutual_info_classif(
        Xt_tr.values,
        y_train.values,
        discrete_features=is_discrete,
        random_state=SEED,
    )
    mi_series = pd.Series(mi, index=Xt_tr.columns).sort_values(ascending=False)
    topk_cols = list(mi_series.head(X_all.shape[1]).index)

    X_fe_train = Xt_tr[topk_cols]
    X_fe_val = Xt_va[topk_cols]
    X_fe_test = Xt_te[topk_cols]

    return {
        "raw": DatasetBundle(X_raw_train, y_train.reset_index(drop=True), X_raw_val, y_val.reset_index(drop=True), X_raw_test, y_test.reset_index(drop=True)),
        "fe": DatasetBundle(X_fe_train, y_train.reset_index(drop=True), X_fe_val, y_val.reset_index(drop=True), X_fe_test, y_test.reset_index(drop=True)),
    }
