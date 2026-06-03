"""Feature engineering helpers for the credit default notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

ENGINEERED_FEATURE_COLS = [
    "MAX_PAY_DELAY",
    "AVG_PAY_DELAY",
    "MONTHS_WITH_DELAY",
    "TOTAL_BILL_AMT",
    "AVG_BILL_AMT",
    "MAX_BILL_AMT",
    "TOTAL_PAY_AMT",
    "AVG_PAY_AMT",
    "MAX_PAY_AMT",
    "PAY_TO_BILL_RATIO",
    "LIMIT_UTILIZATION_1",
    "LIMIT_UTILIZATION_AVG",
    "BILL_CHANGE_1_2",
    "BILL_CHANGE_AVG",
]


def add_features(data):
    """Add engineered repayment, billing, payment, and utilization features."""
    data = data.copy()

    data["MAX_PAY_DELAY"] = data[PAY_STATUS_COLS].max(axis=1)
    data["AVG_PAY_DELAY"] = data[PAY_STATUS_COLS].mean(axis=1)
    data["MONTHS_WITH_DELAY"] = (data[PAY_STATUS_COLS] > 0).sum(axis=1)

    data["TOTAL_BILL_AMT"] = data[BILL_COLS].sum(axis=1)
    data["AVG_BILL_AMT"] = data[BILL_COLS].mean(axis=1)
    data["MAX_BILL_AMT"] = data[BILL_COLS].max(axis=1)

    data["TOTAL_PAY_AMT"] = data[PAY_AMT_COLS].sum(axis=1)
    data["AVG_PAY_AMT"] = data[PAY_AMT_COLS].mean(axis=1)
    data["MAX_PAY_AMT"] = data[PAY_AMT_COLS].max(axis=1)

    data["PAY_TO_BILL_RATIO"] = data["TOTAL_PAY_AMT"] / (
        data["TOTAL_BILL_AMT"].abs() + 1
    )
    data["LIMIT_UTILIZATION_1"] = data["BILL_AMT1"] / (data["LIMIT_BAL"] + 1)
    data["LIMIT_UTILIZATION_AVG"] = data["AVG_BILL_AMT"] / (data["LIMIT_BAL"] + 1)

    data["BILL_CHANGE_1_2"] = data["BILL_AMT1"] - data["BILL_AMT2"]
    data["BILL_CHANGE_AVG"] = data[BILL_COLS].diff(axis=1).iloc[:, 1:].mean(axis=1)

    return data


def make_one_hot_encoder():
    """Create a dense one-hot encoder across supported scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_feature_set(
    source_df,
    target_col,
    use_feature_engineering,
    seed,
    categorical_cols=CATEGORICAL_COLS,
    y_dtype=np.float32,
    output_dtype=np.float32,
):
    """Split and preprocess one baseline or engineered feature set."""
    model_df = add_features(source_df) if use_feature_engineering else source_df.copy()
    X_raw = model_df.drop(columns=["ID", target_col])
    y_raw = model_df[target_col].astype(y_dtype)
    categorical_cols = list(categorical_cols)
    numeric_cols = [col for col in X_raw.columns if col not in categorical_cols]

    X_train_raw, X_temp_raw, y_train_split, y_temp_split = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=seed, stratify=y_raw
    )
    X_val_raw, X_test_raw, y_val_split, y_test_split = train_test_split(
        X_temp_raw, y_temp_split, test_size=0.50, random_state=seed, stratify=y_temp_split
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", make_one_hot_encoder(), categorical_cols),
        ]
    )

    return {
        "use_feature_engineering": use_feature_engineering,
        "engineered_cols": list(ENGINEERED_FEATURE_COLS)
        if use_feature_engineering
        else [],
        "preprocess": preprocess,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "X_train": preprocess.fit_transform(X_train_raw).astype(output_dtype),
        "X_val": preprocess.transform(X_val_raw).astype(output_dtype),
        "X_test": preprocess.transform(X_test_raw).astype(output_dtype),
        "y_train": y_train_split,
        "y_val": y_val_split,
        "y_test": y_test_split,
    }


def prepare_feature_sets(
    source_df,
    target_col,
    seed,
    categorical_cols=CATEGORICAL_COLS,
    y_dtype=np.float32,
    output_dtype=np.float32,
):
    """Build comparable baseline and engineered feature sets."""
    return {
        "baseline": prepare_feature_set(
            source_df,
            target_col=target_col,
            use_feature_engineering=False,
            seed=seed,
            categorical_cols=categorical_cols,
            y_dtype=y_dtype,
            output_dtype=output_dtype,
        ),
        "engineered": prepare_feature_set(
            source_df,
            target_col=target_col,
            use_feature_engineering=True,
            seed=seed,
            categorical_cols=categorical_cols,
            y_dtype=y_dtype,
            output_dtype=output_dtype,
        ),
    }
