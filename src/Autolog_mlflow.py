
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os

import mlflow
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner="VikasRathod314", repo_name="MLFlow-Experiment-tracking-In-MLOPS", mlflow=True)

# Set Tracking URI (IMPORTANT)
mlflow.set_tracking_uri("https://dagshub.com/VikasRathod314/MLFlow-Experiment-tracking-In-MLOPS.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 20
n_estimators = 30

mlflow.autolog()
# Set experiment
mlflow.set_experiment("Wine-Classification-Experiment")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_param('max_depth', max_depth)
    # mlflow.log_param('n_estimators', n_estimators)

    print("Accuracy::", accuracy)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts
    # mlflow.log_artifact("Confusion-matrix.png")

    # Log your source .py only if available
    if os.path.isfile(__file__):
        mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Vikas Rathod', "Project": "Wine Classification"})

    # Log the model (registry disabled)
    #mlflow.sklearn.log_model(rf, artifact_path="Random-Forest-Model", registered_model_name=None)
