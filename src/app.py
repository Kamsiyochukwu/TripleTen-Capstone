"""Natural-language interface for the original Olympic medal-classification model."""
from __future__ import annotations
import re, sys
from pathlib import Path
import pickle
try:  # Supports both `python src/app.py` and package imports in tests.
    from src.train import read_config
except ModuleNotFoundError:
    from train import read_config

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("age", "height", "weight", "year", "sex", "season", "sport")
MEDALS = {0: "No Medal", 1: "Gold", 2: "Silver", 3: "Bronze"}

def parse_query_locally(text):
    lowered, values = text.lower(), {}
    patterns = {"age": r"(\d{1,2})\s*(?:years? old|yo)\b", "height": r"(\d{3})\s*cm\b", "weight": r"(\d{2,3}(?:\.\d+)?)\s*(?:kg|kilograms?)\b", "year": r"\b((?:18|19|20)\d{2})\b"}
    for field, pattern in patterns.items():
        found = re.search(pattern, lowered)
        if found: values[field] = float(found.group(1)) if field in ("height", "weight") else int(found.group(1))
    if re.search(r"\b(female|woman|women|girl)\b", lowered): values["sex"] = "F"
    elif re.search(r"\b(male|man|men|boy)\b", lowered): values["sex"] = "M"
    for season in ("Summer", "Winter"):
        if season.lower() in lowered: values["season"] = season
    sports = {"swimmer": "Swimming", "swimming": "Swimming", "athletics": "Athletics", "runner": "Athletics", "basketball": "Basketball", "football": "Football", "gymnastics": "Gymnastics", "judo": "Judo", "cycling": "Cycling", "rowing": "Rowing", "skiing": "Alpine Skiing"}
    for token, sport in sports.items():
        if token in lowered: values["sport"] = sport; break
    return values

def validate_features(values):
    missing = [field for field in REQUIRED if field not in values]
    return not missing, missing

def parse_query(text):
    """Extract model fields locally without any external LLM or API dependency."""
    return parse_query_locally(text)

def predict(values, model=None):
    valid, missing = validate_features(values)
    if not valid: raise ValueError("Missing: " + ", ".join(missing))
    import pandas as pd
    config = read_config()
    if model is None:
        with open(ROOT / config["artifact_path"], "rb") as file:
            model = pickle.load(file)
    row = pd.DataFrame([{ "Age": values["age"], "Height": values["height"], "Weight": values["weight"], "Year": values["year"], "Sex": values["sex"], "Season": values["season"], "Sport": values["sport"] }])
    probabilities = model.predict_proba(row)[0]; classes = model.named_steps["model"].classes_
    return {MEDALS[int(label)]: float(probability) for label, probability in zip(classes, probabilities)}

def respond(text, model=None):
    values = parse_query(text); valid, missing = validate_features(values)
    if not valid: return "I need " + ", ".join(missing) + ". Include age, height in cm, weight in kg, Olympic year, sex, season, and sport."
    try: probabilities = predict(values, model)
    except FileNotFoundError: return "The model artifact is not available yet. Run `python -m src.train`, then try again."
    predicted = max(probabilities, key=probabilities.get)
    return f"The model predicts **{predicted}** ({probabilities[predicted]:.1%} probability). This is a historical classification pattern, not a guarantee; competition, qualification, and event-specific factors are not represented."

def run_streamlit():
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return run_cli()
    st.set_page_config(page_title="Olympic Medal Predictor", page_icon="🏅")
    st.title("🏅 Olympic Medal Predictor")
    st.caption("Predicts No Medal, Gold, Silver, or Bronze from an athlete profile.")
    query = st.text_area("Example", "I am a 24 year old female swimmer, 172 cm and 63 kg, competing in 2016 Summer.")
    if st.button("Predict medal class"): st.markdown(respond(query))

def run_cli():
    """Dependency-free fallback for environments without Streamlit installed."""
    print("Olympic Medal Predictor (terminal mode)")
    print("Describe an athlete, or type 'quit' to exit.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if query.lower() in {"quit", "exit"}:
            print("Goodbye.")
            return
        if query:
            print("\nPredictor:", respond(query))

if __name__ == "__main__":
    # `streamlit run src/app.py` keeps "run" in argv; direct Python execution
    # intentionally uses the dependency-free terminal mode.
    run_streamlit() if "run" in sys.argv else run_cli()
