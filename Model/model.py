""" handles the AlphaZero model
"""
import os
import sys
import pickle
import time
import keras
import numpy as np
import argparse

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.utility import *
from lib.timing import timing

from azts.config import DATASETDIR
from Player.config import Config
from Model.resnet import resnet_model

from lib.logger import get_logger
log = get_logger("Model")

mlflow.set_tracking_uri("http://35.223.113.101:8000")


class AZero:
    """ The AlphaZero Class

    Attributes:
        model (Keras Model): The ResNet Model with two output heads
        initial_epoch

    Functions:
        train: starts the training process
        restore_latest_model: restores newest weights from model directory
        remember_model_architecture: makes sure the architecture along with config is saved once
        build_model: builds model
        plot_model: plots the network graph
        summary: prints config parameters and model summary
    """

    def __init__(self, config):
        """
        Args:
            config (Config): Player Configuration file
        """

        self.config = config
        self.config.dataset_dir = DATASETDIR    # XXX Tell Seba to use dataset_dir

        self.initial_epoch = 0
        self.checkpoint_file = None

        if self.config.model.load_from_mlflow:
            self.load_mlflow()
        else:
            self.build_model()
            self.remember_model_architecture()
            self.restore_latest_model()

        self.compile_model()
        self.setup_callbacks()

    def load_mlflow(self, version=None):
        if version is None:
            version = mlflow_get_latest_version(self.config.name)
            if version is None:
                log.info(
                    "No model registered as %s found on mlflow server.", self.config.name)
                self.build_model()
                return
        model_uri = self.config.model_uri_format.format(version=version)
        log.info("Fetching model %s version %s from mlflow server.",
                 self.config.name, version)
        self.model = mlflow.keras.load_model(model_uri)

    def auto_run_training(self, max_iterations=5, max_epochs=10):
        """ Automatically enters a training loop that fetches newest datasets
        """
        for i in range(max_iterations):
            dataset_file = get_latest_dataset_file(self.config.dataset_dir)
            if dataset_file is None:
                log.info("No dataset found in %s. Waiting..",
                         self.config.dataset_dir)
                while dataset_file is None:
                    time.sleep(1)
                    dataset_file = get_latest_dataset_file(
                        self.config.dataset_dir)
            self.setup_callbacks(auto_run=True)  # new file, new callback
            with open(dataset_file, 'rb') as f:
                train_data = pickle.load(f)
            log.info("Commencing training %i/%i on dataset %s.",
                     i, max_iterations, dataset_file)
            self.train(train_data, epochs=max_epochs)

    # @timing
    def inference(self, input):
        policy, value = self.model.predict(input[None, :])
        return policy.squeeze(), value.squeeze()

    def setup_callbacks(self, auto_run=False):
        # checkpoint_file = os.path.join(self.config.checkpoint_dir,
        #                                "{epoch:02d}-{loss:.2f}.hdf5")
        # checkpoint = ModelCheckpoint(filepath=checkpoint_file,
        #                              # monitor='val_acc',
        #                              save_weights_only=True,
        #                              verbose=1,
        #                              save_best_only=False)

        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        #                                cooldown=0,
        #                                patience=5,
        #                                min_lr=0.5e-6)
        # callbacks = [checkpoint, lr_reducer]

        save_model = LogCallback(self.config.name,
                                 save_model_every=2,
                                 save_local=True,
                                 save_mlflow=True,
                                 local_dir=self.config.checkpoint_dir)
        # epoch = CountEpochsCallback()

        callbacks = [save_model]

        if auto_run:
            auto_fetch_dataset = AutoFetchDatasetCallback(
                self.config.dataset_dir)
            callbacks.append(auto_fetch_dataset)

        self.callbacks = callbacks

    def train(self, train_data, batch_size=64, epochs=10, initial_epoch=None):
        """ Enters the training loop """

        x_train, y_train = prepare_dataset(train_data)

        if initial_epoch is None:
            initial_epoch = self.initial_epoch

        if epochs == -1:  # train indefinitely; XXX: review
            epochs = 10000

        # begin training
        train_logs = self.model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=initial_epoch + epochs,
                                    shuffle=True,
                                    callbacks=self.callbacks,
                                    initial_epoch=initial_epoch,
                                    verbose=2)

        self.initial_epoch = train_logs.history['epoch'][-1] + 1
        # idk wo die config gerade ist. im prinzip loggt man die so
        # mlflow.log_artifact(artifact_path="config", local_path="path/to/config")

    def summary(self):
        """ Prints a summary of the model architecture """
        print("Model Name: " + self.config.model_name)
        print("Configuration Settings:")
        print(self.config)
        self.model.summary()

    def plot_model(self):
        """ Plots the whole model architecture as a graph """
        # graphviz (not a python package) has to be installed https://www.graphviz.org/
        plot_model(self.model, to_file='Model/%s.png' % self.config.model_name,
                   show_shapes=True, show_layer_names=True)

    def remember_model_architecture(self):
        """ Makes sure that the architecture and config file
        are stored once per model version
        """
        # save model
        model_file = os.path.join(
            self.config.checkpoint_dir, "architecture.yaml")
        if not os.path.isfile(model_file):
            with open(model_file, 'w') as f:
                f.write(self.model.to_yaml())

        # save config
        config_file = os.path.join(self.config.checkpoint_dir, "config.yaml")
        if not os.path.isfile(config_file):
            self.config.dump(config_file)

    def load_model_architecture(self, file):
        """ Restores model architecture from yaml file """
        self.model = keras.models.model_from_yaml(file)

    def new_model_available(self):
        """ checks whether a new checkpoint file is available 
        not very robust; only checks for file name """
        new_checkpoint_file, _ = newest_checkpoint_file(
            self.config.checkpoint_dir)
        return not new_checkpoint_file == self.checkpoint_file

    def restore_latest_model(self):
        """ Checks for latest model checkpoint and restores the weights """
        checkpoint_file, self.initial_epoch = newest_checkpoint_file(
            self.config.checkpoint_dir)

        if checkpoint_file is not None:
            self.restore_from_checkpoint(checkpoint_file)
        else:
            log.info("No previous checkpoint found - initializing new network.")

    def restore_from_checkpoint(self, checkpoint_file):
        """ Restores weights from given checkpoint """
        log.info("Restoring from checkpoint %s", checkpoint_file)
        tries = 3
        while tries:
            try:
                self.model.load_weights(checkpoint_file)
                break
            except OSError:
                if tries:
                    log.warning(
                        "Could not open checkpoint. Retrying in 5 seconds..")
                    time.sleep(5)
                    tries -= 1
                else:
                    raise(OSError)

        self.checkpoint_file = checkpoint_file

    def compile_model(self):
        """ Compiles the model """
        losses = {"policy_head": "categorical_crossentropy",
                  "value_head": "mean_squared_error"}
        learning_rate = self.config.model.training.learning_rate
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss=losses,
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def build_model(self):
        """ Builds the ResNet model via config parameters """

        input_shape = self.config.model.input_shape
        input = keras.layers.Input(shape=input_shape)

        policy_head, value_head = resnet_model(input, self.config)
        self.model = keras.models.Model(inputs=[input],
                                        outputs=[policy_head, value_head],
                                        name=self.config.model_name)


class CountEpochsCallback(keras.callbacks.Callback):
    """ keras Callback Class, used to count epochs """

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch"] = epoch


class LogCallback(keras.callbacks.Callback):
    """ keras Callback Class, used to abort training
    if new datasets are available """

    def __init__(self, model_name, log_every=1, save_model_every=1, save_local=True, save_mlflow=True, local_dir="_Data/models"):
        # pylint: disable=bad-super-call
        super(keras.callbacks.Callback, self).__init__()
        self.save_local_fmt = "{epoch:02d}-{loss:.2f}.hdf5"
        self.log_every = log_every
        self.save_model_every = save_model_every
        self.save_local = save_local
        self.save_mlflow = save_mlflow
        self.local_dir = local_dir
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):

        logs["epoch"] = epoch

        if epoch % self.log_every == 0:
            metrics = ["loss", "policy_head_accuracy", "policy_head_loss",
                       "value_head_accuracy", "value_head_loss", "epoch"]
            for m in metrics:
                mlflow.log_metric(m, logs[m], step=epoch)

        if epoch % self.save_model_every == 0:
            if self.save_local:
                file = os.path.join(self.local_dir, self.save_local_fmt.format(
                    epoch=epoch, loss=logs["loss"]))
                log.info("Saving model to %s...", file)
                self.model.save_weights(file)
            if self.save_mlflow:
                log.info("Saving model to mlflow as %s...", self.model_name)
                mlflow.keras.log_model(artifact_path="model",
                                       keras_model=self.model,
                                       keras_module=keras,
                                       registered_model_name=self.model_name)


class AutoFetchDatasetCallback(keras.callbacks.Callback):
    """ keras Callback Class, used to abort training
    if new datasets are available """

    def __init__(self, dataset_dir):
        # pylint: disable=bad-super-call
        super(keras.callbacks.Callback, self).__init__()
        self.dataset_dir = dataset_dir
        self.current_dataset_file = get_latest_dataset_file(dataset_dir)

    def on_train_batch_end(self, batch, logs=None):
        new_dataset_file = get_latest_dataset_file(self.dataset_dir)
        if not self.current_dataset_file == new_dataset_file:   # XXX use mtime instead?
            log.info("New dataset found: %s", new_dataset_file)
            log.info("Aborting training.")
            self.model.stop_training = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model")
    parser.add_argument(
        "--player", type=str, default="Player/default_config.yaml", help="Path to config file")
    parser.add_argument("-i", "--max_iterations",
                        type=int, default=3)
    parser.add_argument("-ep", "--max_epochs", type=int, default=10000)
    args = parser.parse_args()

    config = Config(args.player)

    model = AZero(config)
    # model.summary()
    model.auto_run_training(max_epochs=args.max_epochs,
                            max_iterations=args.max_iterations)
