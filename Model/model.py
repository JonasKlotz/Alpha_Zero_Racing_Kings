""" handles the AlphaZero model
"""
import os
import re
import numpy as np
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

from lib.timing import timing


class AZero:
    """ The AlphaZero Class

    Attributes:
        model (Keras Model): The ResNet Model with two output heads
        initial_epoch

    Functions:
        train: starts the training process
        restore_weights: restores newest weights from model directory
        remember_model_architecture: makes sure the architecture along with config is saved once
        build_model: builds model
        plot_model: plots the network graph
        summary: ouputs config parameters and model summary
    """

    def __init__(self, config):

        assert config is not None, "ERROR! no config provided"

        self.config = config
        self.build_model()
        self.compile_model()
        self.remember_model_architecture()
        self.initial_epoch = 0
        self.restore_weights()

    def train(self, train_data, batch_size=64, epochs=10, initial_epoch=0):
        """ enters the training loop
        """
        # prepare data
        x_train, y_train_p, y_train_v = np.hsplit(np.array(train_data), [1, 2])
        x_train = np.stack(x_train.flatten(), axis=0)
        y_train_p = np.stack(y_train_p.flatten(), axis=0)
        y_train_v = np.stack(y_train_v.flatten(), axis=0)
        y_train = {"policy_head": y_train_p,
                   "value_head":  y_train_v}

        # Callbacks
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
        callbacks = [checkpoint]

        if initial_epoch == 0:
            initial_epoch = self.initial_epoch

        # begin training
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=initial_epoch + epochs,
                       shuffle=True,
                       callbacks=callbacks,
                       initial_epoch=initial_epoch)
        self.initial_epoch = initial_epoch + epochs

    def summary(self):
        """ prints a summary of the model architecture
        """
        print("Model Name: " + self.config.model_name)
        print("Configuration Settings:")
        print(self.config)
        self.model.summary()

    def plot_model(self):
        """ plots the whole model architecture as a graph
        """
        # graphviz (not a python package) has to be installed https://www.graphviz.org/
        plot_model(self.model, to_file='Model/%s.png' % self.config.model_name,
                   show_shapes=True, show_layer_names=True)

    def remember_model_architecture(self):
        """ makes sure that the architecture and config file
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
            self.config.dump_yaml(config_file)

    def restore_weights(self, checkpoint_file=None):
        """ Checks for latest model checkpoint and restores
        unless a checkpoint is given
        """
        if checkpoint_file is None:
            chk_dir = self.config.checkpoint_dir
            self.initial_epoch = 0
            for file_name in reversed(sorted(os.listdir(chk_dir))):
                file = os.path.join(chk_dir, file_name)
                if os.path.isfile(file):
                    reg = re.search("^0*(\d+).*?\.hdf5", file_name)
                    if reg is not None:
                        self.initial_epoch = int(reg.group(1))
                        checkpoint_file = file
                        break

        if checkpoint_file is not None:
            print("restoring from checkpoint " + checkpoint_file)
            self.model.load_weights(checkpoint_file)
        else:
            print("no previous checkpoint found")

    def compile_model(self):
        """ compiles the model
        """
        losses = {"policy_head": "categorical_crossentropy",
                  "value_head": "mean_squared_error"}
        self.model.compile(loss=losses,
                           optimizer=Adam(learning_rate=1e-3),
                           metrics=['accuracy'])

    def build_model(self):
        """ Builds the ResNet model via config parameters
        """

        def residual_layer(input,
                           num_filters,
                           kernel_size=3,
                           stride=1,
                           activation='relu',
                           batch_normalization=True):
            """ A residual layer
            Args:
                input (tensor): input tensor
                num_filters (int): number of convolutional filters
                kernel_size (int): size of the convolutional kernels
                stride (int): step size of the filters
                activation (string): activation function
                batch_normalization (bool): apply batch normalization
            Returns:
                x (tensor): output tensor of residual layer
            """
            _x = input
            _x = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(_x)
            if batch_normalization:
                _x = BatchNormalization()(_x)
            if activation is not None:
                _x = Activation(activation)(_x)
            return _x

        def body_model(self, input):
            # read config
            num_res_blocks = self.config.model.resnet_depth
            num_layers = self.config.model.residual_block.layers
            num_filters = self.config.model.residual_block.num_filters
            filter_size = self.config.model.residual_block.filter_size
            filter_stride = self.config.model.residual_block.filter_stride
            activation = self.config.model.residual_block.activation
            batch_normalization = self.config.model.residual_block.batch_normalization

            def res_layer(input, activation=activation):
                return residual_layer(input=input,
                                      num_filters=num_filters,
                                      kernel_size=filter_size,
                                      stride=filter_stride,
                                      activation=activation,
                                      batch_normalization=batch_normalization)

            # build model
            _x = input
            _x = res_layer(_x)
            for _ in range(num_res_blocks):
                # residual block
                y = _x
                for layer in range(num_layers):
                    if layer == num_layers - 1:
                        # XXX Why does resnet do this?
                        _x = res_layer(_x, activation=None)
                    else:
                        _x = res_layer(_x)
                # skip connection
                _x = keras.layers.add([_x, y])
                _x = Activation(activation)(_x)
            return _x

        def policy_head_model(self, input):
            # read config
            opt = self.config.model.policy_head
            res_num_filters = opt.residual_layer.num_filters
            res_filter_size = opt.residual_layer.filter_size
            res_filter_stride = opt.residual_layer.filter_stride
            res_batch_normalization = opt.residual_layer.batch_normalization
            res_activation = self.config.model.residual_block.activation
            dense_num_filters = opt.dense_layer.num_filters
            dense_activation = opt.dense_layer.activation

            # build model
            _x = input
            _x = residual_layer(input=_x,
                                num_filters=res_num_filters,
                                kernel_size=res_filter_size,
                                stride=res_filter_stride,
                                activation=res_activation,
                                batch_normalization=res_batch_normalization)
            _x = Dense(dense_num_filters,
                       activation=dense_activation,
                       kernel_initializer='he_normal',
                       name="policy_head")(_x)
            return _x

        def value_head_model(self, input):
            # read config
            opt = self.config.model.value_head
            res_num_filters = opt.residual_layer.num_filters
            res_filter_size = opt.residual_layer.filter_size
            res_filter_stride = opt.residual_layer.filter_stride
            res_batch_normalization = opt.residual_layer.batch_normalization
            res_activation = self.config.model.residual_block.activation
            dense_num_filters = opt.dense_layer.num_filters
            dense_activation = opt.dense_layer.activation

            # build model
            _x = input
            _x = residual_layer(input=_x,
                                num_filters=res_num_filters,
                                kernel_size=res_filter_size,
                                stride=res_filter_stride,
                                activation=res_activation,
                                batch_normalization=res_batch_normalization)
            _x = Flatten()(_x)
            _x = Dense(dense_num_filters,
                       activation='relu',
                       kernel_initializer='he_normal')(_x)
            _x = Dense(1,
                       activation=dense_activation,
                       kernel_initializer='he_normal',
                       name="value_head")(_x)
            return _x

        # define input tensor
        input_shape = self.config.model.input_shape
        input = keras.layers.Input(shape=input_shape)

        # build model
        body = body_model(self, input)
        policy_head = policy_head_model(self, body)
        value_head = value_head_model(self, body)

        self.model = keras.models.Model(inputs=[input],
                                        outputs=[policy_head, value_head],
                                        name=self.config.model_name)


if __name__ == "__main__":
    # TEST

    from Player.config import Config
    config = Config("Player/config.yaml")

    model = AZero(config)
    model.summary()
