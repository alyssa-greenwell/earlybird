import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("debug_test")

with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("f1", 0.91)
    print("Logged metrics")

print("Artifacts path exists:", mlflow.get_tracking_uri())
