"""Preprocessing for the original Olympic athlete medal project."""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEDAL_MAPPING = {"No Medal": 0, "Gold": 1, "Silver": 2, "Bronze": 3}
NUMERIC_FEATURES = ["Age", "Height", "Weight", "Year"]
CATEGORICAL_FEATURES = ["Sex", "Season", "Sport"]

def validate_dataframe(df, required_columns, target_column):
    missing = [col for col in required_columns + [target_column] if col not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if df.empty: raise ValueError("Dataframe is empty")
    return True

def clean_data(df, numeric_columns, categorical_columns):
    """Fill missing data on a copy; never mutate the caller's dataframe."""
    cleaned = df.copy(deep=True)
    for col in numeric_columns:
        if col in cleaned: cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in categorical_columns:
        if col in cleaned:
            mode = cleaned[col].mode(dropna=True)
            cleaned[col] = cleaned[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return cleaned

def encode_categoricals(df, columns):
    return pd.get_dummies(df.copy(deep=True), columns=columns, dtype=int)

def check_data_quality(df, numeric_columns):
    return {"total_rows": len(df), "total_nulls": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
            **{f"{col}_min": float(df[col].min()) for col in numeric_columns if col in df}}

def build_preprocessor(numeric_features, categorical_features):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, numeric_features), ("categorical", categorical, categorical_features)])

def load_and_prepare_data(config):
    """Load Olympic data, retain the original four medal classes, and return raw fields."""
    fields = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = pd.read_csv(config["data_url"], usecols=fields + [config["target"]])
    validate_dataframe(df, fields, config["target"])
    if config.get("sample_size") and len(df) > config["sample_size"]:
        df = df.sample(n=config["sample_size"], random_state=config["random_state"])
    y = df[config["target"]].fillna("No Medal").map(MEDAL_MAPPING).astype(int).rename("Medal")
    return df[fields].copy(), y, len(df)
