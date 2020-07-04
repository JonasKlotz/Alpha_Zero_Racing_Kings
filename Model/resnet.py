from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Flatten, Add
from keras.activations import softmax
from keras.regularizers import l2


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


def resnet_model(input, config):
    """ Builds the ResNet model via config parameters """

    def body_model(input):
        # read config
        num_res_blocks = config.model.resnet_depth
        num_layers = config.model.residual_block.layers
        num_filters = config.model.residual_block.num_filters
        filter_size = config.model.residual_block.filter_size
        filter_stride = config.model.residual_block.filter_stride
        activation = config.model.residual_block.activation
        batch_normalization = config.model.residual_block.batch_normalization

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
                    _x = res_layer(_x, activation=None)
                else:
                    _x = res_layer(_x)
            # skip connection
            _x = Add()([_x, y])
            _x = Activation(activation)(_x)
        return _x

    def policy_head_model(input):
        # read config
        opt = config.model.policy_head
        res_num_filters = opt.residual_layer.num_filters
        res_filter_size = opt.residual_layer.filter_size
        res_filter_stride = opt.residual_layer.filter_stride
        res_batch_normalization = opt.residual_layer.batch_normalization
        res_activation = config.model.residual_block.activation
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
                   name="policy_head_logits")(_x)
        return _x

    def value_head_model(input):
        # read config
        opt = config.model.value_head
        res_num_filters = opt.residual_layer.num_filters
        res_filter_size = opt.residual_layer.filter_size
        res_filter_stride = opt.residual_layer.filter_stride
        res_batch_normalization = opt.residual_layer.batch_normalization
        res_activation = config.model.residual_block.activation
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

    # build model
    body = body_model(input)
    policy_head_logits = policy_head_model(body)
    policy_head = softmax(policy_head_logits)
    value_head = value_head_model(body)

    return policy_head_logits, policy_head, value_head
