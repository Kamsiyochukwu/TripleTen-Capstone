"""Programmatically compare original Olympic-model MLflow runs."""
import mlflow
try:
    from src.train import read_config
except ModuleNotFoundError:
    from train import read_config
def best_run():
    config = read_config(); mlflow.set_tracking_uri(config["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(config["experiment_name"])
    if experiment is None: raise RuntimeError("Run `python -m src.train` first.")
    runs = mlflow.search_runs([experiment.experiment_id], order_by=["metrics.auc_roc DESC"])
    if runs.empty: raise RuntimeError("No completed runs found.")
    return runs.iloc[0], runs
if __name__ == "__main__":
    winner, runs = best_run(); print(runs[["tags.mlflow.runName", "metrics.accuracy", "metrics.precision", "metrics.recall", "metrics.f1", "metrics.auc_roc"]].to_string(index=False)); print("\nBest run:", winner["tags.mlflow.runName"])
