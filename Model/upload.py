import os
import sys
import argparse

import mlflow
import mlflow.keras
import keras


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.model import AZero
from Player.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a local model to mlflow")
    parser.add_argument("--player", type=str)

    args = parser.parse_args()

    config = Config(args.player)
    # override for upload
    config.model.logging.load_from_mlflow = False
    config.model.logging.save_local = False
    config.model.logging.save_mlflow = True

    model = AZero(config)

    model_name = config.name
    print(f"Saving model to mlflow as %s..." % model_name)
    mlflow.keras.log_model(artifact_path="model",
                           keras_model=model.model,
                           keras_module=keras,
                           registered_model_name=model_name)
