import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, add
from keras.regularizers import l2


def build_model(cfg_model):
    ''' Builds the ResNet model
    Args:
        cfg_model (dict): a dictionary containing the
                          model configuration settings
    Returns:
        (body, policy_head, value_head) (tuple):
            body 
    '''
    # read model configuration
    input_shape = cfg_model['input_shape']
    num_res_block = cfg_model['resnet_depth']
    num_res_block_layers = cfg_model['residual_block']['layers']
    num_res_block_filters = cfg_model['residual_block']['filters']
    res_block_filter_size = cfg_model['residual_block']['filter_size']
    res_block_filter_stride = cfg_model['residual_block']['filter_stride']
    res_block_batch_normalization = cfg_model['residual_block']['batch_normalization']
    # policy_output_filters corresponds to the number of available moves
    policy_output_filters = cfg_model['policy_head']['output_shape'][-1]

    # Body
    x = Sequential()
    x = keras.layers.Input(input_shape)
    for res_block in range(num_res_block):
        x = residual_block(input=x,
                           layers=num_res_block_layers,
                           num_filters=num_res_block_filters,
                           kernel_size=res_block_filter_size,
                           stride=res_block_filter_stride,
                           batch_normalization=res_block_batch_normalization)
    body = x

    policy_head = policy_head_model(body, policy_output_filters)
    value_head = value_head_model(body)

    # Do these share weights or will training
    # result in training 3 seperate models?
    return (body, policy_head, value_head)


def residual_layer(input,
                   num_filters,
                   kernel_size=3,
                   stride=1,
                   activation='relu',
                   batch_normalization=True):
    ''' A residual layer

    Args:
        input (tensor): input tensor
        num_filters (int): number of convolutional filters
        kernel_size (int): size of the convolutional kernels
        stride (int): step size of the filters
        activation (string): activation function
        batch_normalization (bool): apply batch normalization

    Returns:
        x (tensor): output tensor of residual layer
    '''
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=stride,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))  #is this all we need for regularisation?
    x = input
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation != None:
        x = Activation(activation)(x)
    return x


def residual_block(input,
                   layers,
                   num_filters,
                   kernel_size=3,
                   stride=1,
                   activation='relu',
                   batch_normalization=True):
    ''' A residual block containing multiple residual layers

    Args:
        input (tensor): input tensor
        layers (int): number of residual layers per block
        num_filters (int): number of convolutional filters
        kernel_size (int): size of the convolutional kernels
        stride (int): step size of the filters
        activation (string): activation function
        batch_normalization (bool): apply batch normalization

    Returns:
        x (tensor): output tensor of residual block
    '''
    x = input
    for layer in range(layers):
        if layer == layers - 1:
            activation = None   # XXX Why does resnet do this?
        x = residual_layer(input=x,
                           num_filters=num_filters,
                           kernel_size=kernel_size,
                           stride=stride,
                           activation=activation)
    # skip connection
    y = residual_layer(input=input,
                       num_filters=num_filters,
                       kernel_size=kernel_size,
                       stride=stride,
                       activation=None)
    x = keras.layers.add([x, y])
    return Activation(activation)(x)


# Disclaimer: I don't know what I'm doing here

def policy_head_model(input, output_filters):
    ''' The policy head model

    Args:
        input (tensor): input tensor
        output_filters (int): number of filters for the last
                              convolutional layer, corresponds to
                              number of actions 8x8xN_act

    Returns:
        x (tensor): policy head model
    '''
    x = input
    x = Conv2D(192,
               kernel_size=1,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_filters,
               kernel_size=1,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    x = Activation('softmax')   # Does this have to be a prob. dist.?
    return x


def value_head_model(input):
    ''' The value head model

    Args:
        input (tensor): input tensor

    Returns:
        x (tensor): value head model
    '''
    output_size = 1
    x = input
    x = Conv2D(192,
               kernel_size=1,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(256,
              activation='relu',
              kernel_initializer='he_normal')
    x = Dense(output_size,
              activation='tanh',
              kernel_initializer='he_normal')
    return x
