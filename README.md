# Olympic Medal Predictor

An end-to-end machine-learning application that predicts an Olympic athlete entry's medal class: No Medal, Gold, Silver, or Bronze. It is aimed at sports fans and analysts who want a quick, transparent estimate from an athlete's profile. The app uses the [Olympic Athletes and Results dataset](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) (271,116 athlete-event records); the included CSV is used locally and is ignored by Git.

## What it does

The training workflow retains the original four `Medal` classes, splits raw records before fitting any transformations, median-imputes numeric data, most-frequent-imputes and one-hot encodes categoricals, and scales numerical columns. It compares Logistic Regression, Random Forest, Gradient Boosting, a neural network, and a tuned Random Forest. Every run logs parameters, data description, metrics, and a model artifact to MLflow.

The Streamlit interface accepts plain English such as: “I am a 24 year old female swimmer, 172 cm and 63 kg, competing in 2016 Summer.” It extracts and validates the model fields locally, invokes the saved sklearn pipeline, and gives a class probability with limitations. No API key or external LLM is required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.train
streamlit run src/app.py
```

Put `athlete_events.csv` in `data/` before training. No API key is required. Start the MLflow dashboard with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Workflow

1. Train and log the model experiments:

   ```bash
   python -m src.train
   ```

   This evaluates the configured model types, logs each run to MLflow, selects the best weighted ROC-AUC, and saves it as `models/best_medal_model.plk`.

2. Compare the tracked experiments:

   ```bash
   python -m src.evaluate
   ```

3. Use the application:

   ```bash
   streamlit run src/app.py
   ```

   If Streamlit is not installed, run `python src/app.py` for the terminal interface instead.

4. Validate changes before committing:

   ```bash
   pytest tests/ -v
   ```

## Architecture

`Natural-language query → parser/LLM → feature validation → saved sklearn pipeline → probability → LLM/local explanation`

`src/train.py` writes the selected pipeline to `models/best_medal_model.plk`. The pipeline contains preprocessing and the classifier, so inference applies exactly the transformations learned from the training data.

## Results

Run `python -m src.train` to generate the final four-class results. `python -m src.evaluate` calls `mlflow.search_runs()` and ranks all five configurations by weighted one-vs-rest ROC-AUC. Accuracy, weighted precision, weighted recall, weighted F1, and ROC-AUC are logged. Medal classes are heavily imbalanced, so the weighted metrics must be interpreted alongside the class distribution; this estimate is not a causal assessment of an athlete.

## Tests and reflection

Run `pytest tests/ -v`. The suite covers missing-value handling, categorical encoding, scaling, immutability, model output/performance, parsing, and incomplete input. The challenging part was balancing a useful conversational interface with clear uncertainty: the app asks for missing values instead of inventing them. With more time, I would add temporal validation, calibrate probabilities, and incorporate only pre-event features.
