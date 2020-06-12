import os

from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    log_param("param1", 17)

    log_metric("foo", 0)
    log_metric("foo", 1)
    log_metric("foo", 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    log_artifacts("outputs")
