import mlflow

if __name__ == "__main__":
	print("Running mlflow_tracking.py")
	with mlflow.start_run():
		mlflow.log_param("param1", 0)
		mlflow.log_metric("metric", 1)
		mlflow.log_artifact("newtest.txt", "newtest")

