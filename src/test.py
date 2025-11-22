import mlflow
#http or https link for tracking the experiments
print(mlflow.__version__)
print("old tracking uri",mlflow.get_tracking_uri())
print("\n\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("New tracking uri ",mlflow.get_tracking_uri())