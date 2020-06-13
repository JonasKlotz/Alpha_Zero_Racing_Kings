""" handles the AlphaZero model
"""
import os
import sys
import re
import pickle
import time
import keras
import mlflow
import mlflow.keras
import numpy as np
import argparse

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

# add root folder to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# relative imports
from lib.timing import timing

# from azts.config import DATASETDIR
from Player.config import Config
from Model.resnet import resnet_model

from lib.logger import get_logger
log = get_logger("Model")


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

        self.initial_epoch = 0
        self.checkpoint_file = None

        if self.config.model.load_from_mlflow:
            self.load_mlflow()  # XXX are all parameters set?
        else:
            self.build_model()
            self.remember_model_architecture()
            self.restore_latest_model()

        self.compile_model()
        self.setup_callbacks()

    def load_mlflow(self, version=None):
        if version is None:
            model_uri = self.config.model_uri
        else:
            model_uri = self.config.model_uri_format.format(version)
        log.info("Loading model from mlflow server.")
        self.model = mlflow.keras.load_model(model_uri)

    def auto_run_training(self, max_iterations=5, max_epochs=10):
        """ Automatically enters a training loop that fetches newest datasets
        """

        for i in range(max_iterations):
            dataset_file = self.get_latest_dataset_file()
            if dataset_file is None:
                log.info("No dataset found in %s. Waiting..",
                         self.config.dataset_dir)
                while dataset_file is None:
                    time.sleep(1)
                    dataset_file = self.get_latest_dataset_file()
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
        checkpoint_file = os.path.join(self.config.checkpoint_dir,
                                       "{epoch:02d}-{loss:.2f}.hdf5")
        checkpoint = ModelCheckpoint(filepath=checkpoint_file,
                                     # monitor='val_acc',
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=False)

        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        #                                cooldown=0,
        #                                patience=5,
        #                                min_lr=0.5e-6)
        # callbacks = [checkpoint, lr_reducer]

        epoch = CountEpochs()

        callbacks = [checkpoint, epoch]

        if auto_run:
            auto_fetch_dataset = AutoFetchDataset(self)
            callbacks.append(auto_fetch_dataset)

        self.callbacks = callbacks

    def train(self, train_data, batch_size=64, epochs=10, initial_epoch=None):
        """ Enters the training loop """

        x_train, y_train = self.prepare_dataset(train_data)

        if initial_epoch is None:
            initial_epoch = self.initial_epoch

        if epochs == -1:  # train indefinitely; XXX: review
            epochs = 10000

        # begin training
        mlflow.set_tracking_uri = "http://35.223.113.101:8000"
        with mlflow.start_run():
            train_logs = self.model.fit(x_train, y_train,
                                        batch_size=batch_size,
                                        epochs=initial_epoch + epochs,
                                        shuffle=True,
                                        callbacks=self.callbacks,
                                        initial_epoch=initial_epoch,
                                        verbose=2)
            self.initial_epoch = train_logs.history['epoch'][-1] + 1

            # mlflow logging

            mlflow.log_param("epochs", epochs)
            # TODO: wo kriegen wir den echten loss her?
            #mlflow.log_metric("loss", 5)
            mlflow.keras.log_model(artifact_path="model",
                                   keras_model=self.model,
                                   keras_module=keras,
                                   registered_model_name=self.config.model_name)
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
        new_checkpoint_file, _ = self.newest_checkpoint_file()
        return not new_checkpoint_file == self.checkpoint_file

    def newest_checkpoint_file(self):
        """ searches checkpoint dir for newest checkpoint,
        goes by file mtime..
        Returns:
            file (str): filepath
            epoch (int): checkpoint's epoch
        """
        chkpt_dir = self.config.checkpoint_dir

        def chkpt_sort(file):   # XXX go for newest epoch?
            return os.path.getmtime(os.path.join(chkpt_dir, file))

        files = reversed(sorted(os.listdir(chkpt_dir), key=chkpt_sort))

        for file_name in files:
            file = os.path.join(chkpt_dir, file_name)
            if os.path.isfile(file):
                reg = re.search("^0*(\d+).*?\.hdf5", file_name)
                if reg is not None:
                    epoch = int(reg.group(1))
                    return file, epoch
        return None, 0

    def restore_latest_model(self):
        """ Checks for latest model checkpoint and restores the weights """
        checkpoint_file, self.initial_epoch = self.newest_checkpoint_file()

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

    def get_latest_dataset_file(self):
        """ Returns newest dataset file in game dir """
        _dir = self.config.dataset_dir
        files = os.listdir(_dir)
        if len(files) == 0:
            return None

        def key_map(file):
            return os.path.getmtime(os.path.join(_dir, file))
        newest_file = max(files, key=key_map)
        return os.path.join(_dir, newest_file)

    def prepare_dataset(self, train_data):
        """ Transforms dataset to format that keras.model expects """
        x_train, y_train_p, y_train_v = np.hsplit(np.array(train_data), [1, 2])
        x_train = np.stack(x_train.flatten(), axis=0)
        y_train_p = np.stack(y_train_p.flatten(), axis=0)
        y_train_v = np.stack(y_train_v.flatten(), axis=0)
        y_train = {"policy_head": y_train_p,
                   "value_head": y_train_v}
        return x_train, y_train


class CountEpochs(keras.callbacks.Callback):
    """ keras Callback Class, used to count epochs """

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch"] = epoch


class AutoFetchDataset(keras.callbacks.Callback):
    """ keras Callback Class, used to abort training
    if new datasets are available """

    def __init__(self, azero):
        # pylint: disable=bad-super-call
        super(keras.callbacks.Callback, self).__init__()
        # pylint: enable=bad-super-call
        self.current_dataset_file = azero.get_latest_dataset_file()
        self.azero = azero

    def on_train_batch_end(self, batch, logs=None):
        new_dataset_file = self.azero.get_latest_dataset_file()
        if not self.current_dataset_file == new_dataset_file:   # XXX use mtime instead?
            log.info("New dataset found: %s", new_dataset_file)
            log.info("Aborting training.")
            self.model.stop_training = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model")
    parser.add_argument(
        "player", type=str, default="Player/default_config.yaml", help="Path to config file")
    parser.add_argument("-i", "--max_iterations",
                        type=int, default=3)
    parser.add_argument("-ep", "--max_epochs", type=int, default=10000)
    args = parser.parse_args()

    config = Config(args.player)

    model = AZero(config)
    # model.summary()
    model.auto_run_training(max_epochs=args.max_epochs,
                            max_iterations=args.max_iterations)

