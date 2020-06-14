import os

from mlflow import log_metric, log_param, log_artifact, start_run, set_tracking_uri,set_experiment, create_experiment

if __name__ == "__main__":
    print("Running mlflow_tracking.py")
    with start_run():
        log_param("param1", 0)
        log_metric("metric", 1)

        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open("outputs/test.txt", "w") as f:
            f.write("hello world!")

        log_artifact("newtest.txt", "newtest.txt")
