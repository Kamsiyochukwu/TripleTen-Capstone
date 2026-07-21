import pandas as pd
from src.preprocess import build_preprocessor, clean_data, encode_categoricals


def sample(): return pd.DataFrame({"Age": [20., None, 30.], "Sport": ["Swimming", None, "Judo"]})
def test_missing_numeric_is_filled(): assert clean_data(sample(), ["Age"], ["Sport"])["Age"].isna().sum() == 0
def test_missing_category_is_filled(): assert clean_data(sample(), ["Age"], ["Sport"])["Sport"].isna().sum() == 0
def test_encoding_creates_columns(): assert any(c.startswith("Sport_") for c in encode_categoricals(sample(), ["Sport"]).columns)
def test_cleaning_does_not_mutate_original():
    df = sample(); clean_data(df, ["Age"], ["Sport"]); assert df.isna().sum().sum() == 2
def test_numeric_features_are_scaled():
    df = clean_data(sample(), ["Age"], ["Sport"]); transformed = build_preprocessor(["Age"], ["Sport"]).fit_transform(df); assert abs(transformed[:, 0].mean()) < 1e-9
