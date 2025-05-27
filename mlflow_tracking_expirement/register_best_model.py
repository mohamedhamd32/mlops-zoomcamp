import os
import pickle
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Constants
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# Set MLflow tracking
mlflow.set_tracking_uri("https://mlflow-q10f8q.adv-cml01.apps.advocprdc.stc.com.sa")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params, X_test, y_test):
    """Train RF on train/val, log explicitly to MLflow"""
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run() as run:
        # Extract hyperparameters
        new_params = {param: int(params[param]) for param in RF_PARAMS}

        # Train model
        rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,  # or adjust to 2/4 if needed
        max_features="sqrt"
        )
        rf.fit(X_train, y_train)

        # Predict and evaluate
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))

        # Log hyperparameters manually
        for param_name, param_value in new_params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model manually
        mlflow.sklearn.log_model(rf, artifact_path="model")


def run_register_model(data_path: str, top_n: int):
    """Select top N HPO runs, retrain, log, register best"""
    client = MlflowClient()

    # Step 1: Get top N models from hyperparameter tuning experiment
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    top_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Step 2: Load test data once for reuse
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Step 3: Train and log top N models using test data
    for run in top_runs:
        train_and_log_model(data_path=data_path, params=run.data.params, X_test=X_test, y_test=y_test)

    # Step 4: Pick the best model by test RMSE
    new_exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=new_exp.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Step 5: Register best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name="random-forest-regressor-best",
        await_registration_for=60  # optional: wait for registration to complete
    )

    print(f"✅ Model registered from run {best_run.info.run_id}")
    print(f"📉 Test RMSE: {best_run.data.metrics['test_rmse']:.4f}")


if __name__ == '__main__':
    run_register_model("/home/cdsw/my_codes/output", 5)
