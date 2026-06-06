"""Feature engineering helpers for the credit default notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "default_of_credit_card_clients.xls"
DEFAULT_ENGINEERED_OUTPUT_PATH = DATA_DIR / "default_of_credit_card_clients_engineered.xlsx"
DEFAULT_TARGET_COL = "default payment next month"

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


def read_credit_dataset(path=DEFAULT_INPUT_PATH, excel_header=1):
    """Read the credit dataset from Excel or CSV."""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path, header=excel_header)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def excel_metadata_columns(columns, target_col=DEFAULT_TARGET_COL):
    """Build the first metadata row used by the original credit Excel file."""
    feature_idx = 1
    metadata_cols = []
    for column in columns:
        if column == "ID":
            metadata_cols.append("")
        elif column == target_col:
            metadata_cols.append("Y")
        else:
            metadata_cols.append(f"X{feature_idx}")
            feature_idx += 1
    return metadata_cols


def order_like_credit_excel(data, target_col=DEFAULT_TARGET_COL):
    """Keep ID first and move the target column to the end, as in the original file."""
    if target_col not in data.columns:
        return data
    non_target_cols = [col for col in data.columns if col != target_col]
    return data[[*non_target_cols, target_col]]


def write_credit_dataset(data, path, target_col=DEFAULT_TARGET_COL):
    """Write credit data to CSV or Excel with the original two-row Excel layout."""
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        data.to_csv(path, index=False)
        return path
    if suffix == ".xlsx":
        metadata = pd.DataFrame([excel_metadata_columns(data.columns, target_col)])
        metadata.to_excel(path, index=False, header=False)
        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            data.to_excel(writer, index=False, startrow=1)
        return path
    if suffix == ".xls":
        raise ValueError(
            "Writing legacy .xls files is not supported by the installed pandas "
            "environment. Use .xlsx for the same Excel layout, or install a legacy "
            ".xls writer and pass a custom output_path."
        )
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def save_engineered_dataset(
    source_df=None,
    input_path=DEFAULT_INPUT_PATH,
    output_path=DEFAULT_ENGINEERED_OUTPUT_PATH,
    excel_header=1,
):
    """Create and save the original dataset plus engineered features."""
    output_path = Path(output_path)
    data = (
        read_credit_dataset(input_path, excel_header)
        if source_df is None
        else source_df
    )
    engineered_data = order_like_credit_excel(add_features(data))

    write_credit_dataset(engineered_data, output_path)
    return engineered_data


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


if __name__ == "__main__":
    engineered_df = save_engineered_dataset()
    print(
        f"Saved engineered dataset to {DEFAULT_ENGINEERED_OUTPUT_PATH} "
        f"with shape {engineered_df.shape}"
    )
