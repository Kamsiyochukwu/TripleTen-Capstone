from sklearn.model_selection import train_test_split
from src.train import build_pipeline, calculate_metrics, read_config
from src.preprocess import load_and_prepare_data


def fitted_model():
    config = read_config(); config["sample_size"] = 1500
    X, y, _ = load_and_prepare_data(config); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    return build_pipeline(config, "logistic_regression").fit(Xtr, ytr), Xte, yte
def test_prediction_shape_and_type():
    model, X, _ = fitted_model(); predictions = model.predict(X); assert predictions.shape == (len(X),) and set(predictions).issubset({0, 1, 2, 3})
def test_model_has_better_than_random_auc():
    model, X, y = fitted_model(); assert calculate_metrics(model, X, y)["auc_roc"] > .50
