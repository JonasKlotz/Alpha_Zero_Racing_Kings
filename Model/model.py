import os
import numpy as np
import keras
from time import time
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model


class AZero:
    """ The AlphaZero Class

    Attributes:
        model (Keras Model): The ResNet Model with two output heads

    Functions:
        read_config: reads in config file and builds model
        build_model: builds model
        summary: ouput config parameters and model summary
        plot_model: plot the network graph
        save_model:
        load_model:
    """

    def __init__(self, config):

        assert config is not None, "ERROR! no config provided"

        self.config = config
        self.build_model()
        self.compile_model()

    def compile_model(self):
        losses = {"policy_head": "categorical_crossentropy",
                  "value_head": "mean_squared_error"}
        self.model.compile(loss=losses,
                           optimizer=Adam(learning_rate=1e-3),
                           metrics=['accuracy'])

    def train(self, train_data):
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
                                     monitor='val_acc',
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=False)

        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        #                                cooldown=0,
        #                                patience=5,
        #                                min_lr=0.5e-6)
        # callbacks = [checkpoint, lr_reducer]
        callbacks = [checkpoint]

        # begin training
        self.model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=120,
                       shuffle=True,
                       callbacks=callbacks)

    def build_model(self):

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
            x = input
            x = Conv2D(num_filters,
                       kernel_size=kernel_size,
                       strides=stride,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation != None:
                x = Activation(activation)(x)
            return x

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
            x = input
            x = res_layer(x)
            for _ in range(num_res_blocks):
                # residual block
                y = x
                for layer in range(num_layers):
                    if layer == num_layers - 1:
                        # XXX Why does resnet do this?
                        x = res_layer(x, activation=None)
                    else:
                        x = res_layer(x)
                # skip connection
                x = keras.layers.add([x, y])
                x = Activation(activation)(x)
            return x

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
            x = input
            x = residual_layer(input=x,
                               num_filters=res_num_filters,
                               kernel_size=res_filter_size,
                               stride=res_filter_stride,
                               activation=res_activation,
                               batch_normalization=res_batch_normalization)
            x = Dense(dense_num_filters,
                      activation=dense_activation,
                      kernel_initializer='he_normal',
                      name="policy_head")(x)
            return x

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
            x = input
            x = residual_layer(input=x,
                               num_filters=res_num_filters,
                               kernel_size=res_filter_size,
                               stride=res_filter_stride,
                               activation=res_activation,
                               batch_normalization=res_batch_normalization)
            x = Flatten()(x)
            x = Dense(dense_num_filters,
                      activation='relu',
                      kernel_initializer='he_normal')(x)
            x = Dense(1,
                      activation=dense_activation,
                      kernel_initializer='he_normal',
                      name="value_head")(x)
            return x

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

    def summary(self):
        print("Model Name: " + self.config.model_name)
        print("Configuration Settings:")
        print(self.config)
        self.model.summary()

    def plot_model(self):
        # graphviz (not a python package) has to be installed https://www.graphviz.org/
        plot_model(self.model, to_file='Model/%s.png' % self.config.model_name,
                   show_shapes=True, show_layer_names=True)

    def save_model(self):  # (self, conf_path, weight_path):  Überlegen ob nur modell oder modell und weights save/load
        """
        :param conf_path: path to save configuration from
        :param weight_path: path to save weights from
        """
        file_name = time() + ".h5"
        path = os.path.join("Model", "Checkpoints", file_name)
        # logger.debug(f"save model to {config_path}")
        print("saving model to " + path)
        self.model.save(path)
        print("sucess")

    def load_model(self, path):  # , conf_path, weight_path):
        """
        :param conf_path: path to load configuration from
        :param weight_path: path to load weights from
        """
        print("loading model " + path)
        # logger.debug(f"load model to {config_path}")
        self.model = keras.models.load_model(path)
        print("sucess")
