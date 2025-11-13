# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for Hugging Face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, Repository
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# MLflow setup
#mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("predictive-maintainance-experiment")

api = HfApi()


api = HfApi(token=os.getenv("HF_TOKEN"))

# Load datasets from Hugging Face
Xtrain_path = "hf://datasets/sudha1726/predictive-maintainanace/Xtrain.csv"
Xtest_path = "hf://datasets/sudha1726/predictive-maintainanace/Xtest.csv"
ytrain_path = "hf://datasets/sudha1726/predictive-maintainanace/ytrain.csv"
ytest_path = "hf://datasets/sudha1726/predictive-maintainanace/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)



# Preprocessing
# Identify numeric and categorical features
numeric_features =  ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]


# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)

)
# Model Definition
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 150],
    'xgbclassifier__max_depth': [4, 5, 6],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.8, 0.9],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.5, 1.0, 1.5]
}

#Model Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Experimentation and Tracking
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)


    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_predictive_maintainance_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # ---------------------------
    # Upload Model to Hugging Face
    repo_id = "sudha1726/predictive_maintainance_model"
    repo_type = "model"
    local_repo_path = "predictive_maintainance_model_repo"

    # Clone the repository or create it if it doesn't exist
    try:
        repo = Repository(local_repo_path, clone_from=repo_id, use_auth_token=os.getenv("HF_TOKEN"))
        print(f"Cloned existing repository: {repo_id}")
    except RepositoryNotFoundError:
        print(f"Repository '{repo_id}' not found. Creating it...")
        repo = Repository(local_repo_path, repo_id=repo_id, repo_type=repo_type, use_auth_token=os.getenv("HF_TOKEN"))
        print(f"Created new repository: {repo_id}")

    # Copy the model file to the local repository
    os.makedirs(local_repo_path, exist_ok=True)
    import shutil
    shutil.copyfile(model_path, os.path.join(local_repo_path, os.path.basename(model_path)))

    # Add changes to the repository
    repo.git_add(f"{os.path.basename(model_path)}")

    # Commit changes
    repo.git_commit(f"Add {os.path.basename(model_path)}")

    # Push changes to the remote repository
    repo.git_push()

    print(f"Model uploaded to Hugging Face repository: {repo_id}")
