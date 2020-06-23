""" handles the AlphaZero model
"""
import os
import sys
import pickle
import time
import argparse

import mlflow
import mlflow.keras
# import tensorflow as tf
# import tensorflow.keras as keras
import keras

from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.utility import *
from lib.timing import timing

from Azts.config import DATASETDIR
from Player.config import Config
from Model.resnet import resnet_model

from lib.logger import get_logger
log = get_logger("Model")

DEBUG = True


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

        self.load_model()
        self.remember_model_architecture()
        self.compile_model()
        self.setup_callbacks()

        if self.config.model.logging.log_mlflow:
            mlflow.log_param("resnet_depth", self.config.model.resnet_depth)
            mlflow.log_param(
                "learning_rate", self.config.model.training.learning_rate)

    def auto_run_training(self, max_iterations=5, max_epochs=10):
        """ Automatically enters a training loop that fetches newest datasets
        """
        for i in range(max_iterations - 1):
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
        policy = policy.squeeze()
        value = value.squeeze()
        if DEBUG:
            if not valid_ndarray(input):
                log.critical("INVALID TENSOR FOUND IN INPUT")
            if not valid_ndarray(policy):
                log.critical("INVALID TENSOR FOUND IN POLICY OUTPUT")
            if not valid_ndarray(value):
                log.critical("INVALID TENSOR FOUND IN VALUE OUTPUT")
        return policy, value

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

        save_model = LogCallback(self.config)

        callbacks = [save_model]

        if auto_run:
            auto_fetch_dataset = AutoFetchDatasetCallback(
                self.config.dataset_dir)
            callbacks.append(auto_fetch_dataset)

        self.callbacks = callbacks

    def train(self, train_data, batch_size=64, epochs=10, initial_epoch=None):
        """ Enters the training loop """

        x_train, y_train = prepare_dataset(train_data)

        if DEBUG:
            assert valid_ndarray(x_train), "INVALID ndarray FOUND IN x_train"
            assert valid_ndarray(
                y_train["policy_head"]), "INVALID ndarray FOUND IN policy of y_train"
            assert valid_ndarray(
                y_train["value_head"]), "INVALID ndarray FOUND IN value of y_train"

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
        # if config.model.logging.save_mlflow:
        #     return  # XXX implement?

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

    def restore_local_model(self):
        """ Checks for latest model checkpoint and restores the weights """
        checkpoint_file, self.initial_epoch = newest_checkpoint_file(
            self.config.checkpoint_dir)

        if checkpoint_file is not None:
            self.restore_from_checkpoint(checkpoint_file)
        else:
            log.info("No previous checkpoint found.")
            log.info("Initializing new network.")

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
        categorical = keras.losses.CategoricalCrossentropy(from_logits=True)
        losses = {"policy_head": categorical,
                  "value_head": "mean_squared_error"}
        learning_rate = self.config.model.training.learning_rate
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(loss=losses,
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def load_model(self):
        if self.config.model.logging.load_from_mlflow:
            self.load_from_mlflow(
                self.config.model.logging.mlflow_model_version)
        else:
            self.load_local_model()

    def load_local_model(self):
        self.build_model()
        self.restore_local_model()

    def load_from_mlflow(self, version=0):
        if version is 0:    # search for newest model on server
            version = mlflow_get_latest_version(self.config.name)
            if version is 0:    # no model version registered
                log.info(
                    "No model registered as %s found on mlflow server.", self.config.name)
                log.info("Initializing new network.")
                self.build_model()
                return
        model_uri = self.config.model_uri_format.format(version=version)
        log.info("Fetching model %s version %s from mlflow server.",
                 self.config.name, version)
        self.model = mlflow.keras.load_model(model_uri)

    def build_model(self):
        """ Builds the ResNet model via config parameters """

        input_shape = self.config.model.input_shape
        input = keras.layers.Input(shape=input_shape)

        policy_head, value_head = resnet_model(input, self.config)
        self.model = keras.models.Model(inputs=[input],
                                        outputs=[policy_head, value_head],
                                        name=self.config.model_name)


class LogCallback(keras.callbacks.Callback):
    """ keras Callback Class, used to log metrics """

    def __init__(self, config):
        # pylint: disable=bad-super-call
        super(keras.callbacks.Callback, self).__init__()
        self.config = config
        self.metrics = ["loss", "policy_head_accuracy", "policy_head_loss",
                        "value_head_accuracy", "value_head_loss", "epoch"]

    def on_epoch_end(self, epoch, logs=None):

        logs["epoch"] = epoch

        log_every = self.config.model.logging.log_metrics_every
        log_mlflow = self.config.model.logging.log_mlflow
        save_model_every = self.config.model.logging.save_model_every
        save_local = self.config.model.logging.save_local
        save_mlflow = self.config.model.logging.save_mlflow
        local_dir = self.config.checkpoint_dir
        model_name = self.config.name

        save_local_fmt = "{epoch:02d}-{loss:.2f}.hdf5"

        if log_mlflow and epoch % log_every == 0:
            for metric in self.metrics:
                mlflow.log_metric(metric, logs[metric], step=epoch)

        if (epoch + 1) % save_model_every == 0:
            if save_local:
                file = os.path.join(local_dir, save_local_fmt.format(
                    epoch=epoch, loss=logs["loss"]))
                log.info("Saving model to %s...", file)
                self.model.save_weights(file)
            if save_mlflow:
                log.info("Saving model to mlflow as %s...", model_name)
                mlflow.keras.log_model(artifact_path="model",
                                       keras_model=self.model,
                                       keras_module=keras,
                                       registered_model_name=model_name)

    def on_train_batch_end(self, batch, logs=None):
        if DEBUG:
            log.debug(logs)
            valid = True
            for l, layer in enumerate(self.model.layers):
                weights = layer.get_weights()
                if l == 0:
                    continue
                for i, w in enumerate(weights):
                    if not valid_ndarray(w):
                        log.critical(
                            "Invalid entry (%i) found in layer %s (%i)", i, layer.name, l)
                        valid = False

            assert valid, "Invalid layer"


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
        if not self.current_dataset_file == new_dataset_file:
            log.info("New dataset found: %s", new_dataset_file)
            log.info("Aborting training.")
            self.model.stop_training = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model")
    parser.add_argument(
        "--player", type=str, default="Player/default_config.yaml", help="Path to config file")
    parser.add_argument("-i", "--max_iterations", type=int, default=3)
    parser.add_argument("-ep", "--max_epochs", type=int, default=10000)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    DEBUG = args.debug

    config = Config(args.player)

    model = AZero(config)
    # model.summary()
    # model.plot_model()
    model.auto_run_training(max_epochs=args.max_epochs,
                            max_iterations=args.max_iterations)
