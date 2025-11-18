import mlflow
import mlflow.sklearn


def init_experiment(experiment_name="ledgerx_failure_model"):
    mlflow.set_experiment(experiment_name)


def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_pipeline(model_name: str, pipeline):
    mlflow.sklearn.log_model(pipeline, model_name)
