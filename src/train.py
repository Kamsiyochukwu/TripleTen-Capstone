"""Train the original four-class Olympic medal models with MLflow."""
from __future__ import annotations
from pathlib import Path
import joblib, yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
try:  # Supports both `python -m src.train` and `python src/train.py`.
    from src.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, load_and_prepare_data
except ModuleNotFoundError:
    from preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, load_and_prepare_data

ROOT = Path(__file__).resolve().parents[1]
def read_config():
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as file: return yaml.safe_load(file)

def make_estimator(model_type, config):
    state = config["random_state"]
    if model_type == "logistic_regression": return LogisticRegression(C=config["lr_C"], max_iter=1000, class_weight="balanced", random_state=state)
    if model_type == "random_forest": return RandomForestClassifier(n_estimators=config["rf_n_estimators"], max_depth=config["rf_max_depth"], class_weight="balanced", n_jobs=-1, random_state=state)
    if model_type == "random_forest_tuned": return RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=state)
    if model_type == "gradient_boosting": return GradientBoostingClassifier(n_estimators=config["gb_n_estimators"], learning_rate=config["gb_learning_rate"], max_depth=config["gb_max_depth"], random_state=state)
    if model_type == "neural_network": return MLPClassifier(hidden_layer_sizes=tuple(config["nn_hidden_layer_sizes"]), activation=config["nn_activation"], solver=config["nn_solver"], early_stopping=True, max_iter=250, random_state=state)
    raise ValueError(f"Unknown model type: {model_type}")

def build_pipeline(config, model_type):
    return Pipeline([("preprocess", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)), ("model", make_estimator(model_type, config))])

def _train_for_type(config, model_type):
    X, y, n_rows = load_and_prepare_data(config)
    # Your original config stores several test splits. The final split is used for
    # this model return value, matching the original function flow without leakage.
    test_size = config["test_size"][-1] if isinstance(config["test_size"], list) else config["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=config["random_state"], stratify=y)
    model = build_pipeline(config, model_type).fit(X_train, y_train)
    return model, X_test, y_test, test_size

# Original project function names and return format are retained.
def train_model_LR(config): return _train_for_type(config, "logistic_regression")
def train_model_RF(config): return _train_for_type(config, "random_forest")
def train_model_GB(config): return _train_for_type(config, "gradient_boosting")
def train_model_NN(config): return _train_for_type(config, "neural_network")

def calculate_metrics(model, X_test, y_test):
    prediction, probability = model.predict(X_test), model.predict_proba(X_test)
    return {"accuracy": accuracy_score(y_test, prediction), "precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "recall": recall_score(y_test, prediction, average="weighted", zero_division=0), "f1": f1_score(y_test, prediction, average="weighted", zero_division=0),
            "auc_roc": roc_auc_score(y_test, probability, multi_class="ovr", average="weighted")}

def train_all(config=None):
    import mlflow, mlflow.sklearn
    config = config or read_config(); mlflow.set_tracking_uri(config["tracking_uri"]); mlflow.set_experiment(config["experiment_name"])
    results = []
    for model_type in config["model_type"]:
        with mlflow.start_run(run_name=model_type) as run:
            model, X_test, y_test, test_size = _train_for_type(config, model_type)
            metrics = calculate_metrics(model, X_test, y_test)
            mlflow.log_params({"model_type": model_type, "data_rows": len(X_test) / test_size, "data_version": "athlete_events.csv", "test_size": test_size, "handle_missing": config["handle_missing"], "scale_features": config["scale_features"]})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
            results.append({"name": model_type, "run_id": run.info.run_id, "metrics": metrics, "model": model})
            print(f"{model_type}: " + ", ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
    best = max(results, key=lambda result: result["metrics"]["auc_roc"])
    path = ROOT / config["artifact_path"]; path.parent.mkdir(exist_ok=True); joblib.dump(best["model"], path)
    print(f"Best model: {best['name']} → {path}"); return results

if __name__ == "__main__": train_all()
