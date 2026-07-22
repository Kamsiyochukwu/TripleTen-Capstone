"""Provide a small local dataset only when the ignored source CSV is unavailable.

This keeps the test suite runnable in clean CI clones without committing the
full Olympic dataset.
"""
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def test_dataset():
    path = Path(__file__).resolve().parents[1] / "data" / "athlete_events.csv"
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True)
    sports = ["Swimming", "Athletics", "Judo", "Basketball"]
    medals = [None, "Gold", "Silver", "Bronze"]
    rows = []
    for index in range(240):
        medal_index = index % 4
        rows.append({"Age": 20 + medal_index * 4 + index % 3, "Height": 160 + medal_index * 7,
                     "Weight": 55 + medal_index * 8, "Year": 2000 + index % 20,
                     "Sex": "F" if medal_index % 2 else "M", "Season": "Summer",
                     "Sport": sports[medal_index], "Medal": medals[medal_index]})
    pd.DataFrame(rows).to_csv(path, index=False)
