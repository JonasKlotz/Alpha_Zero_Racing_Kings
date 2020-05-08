#!/usr/bin/env python3

import yaml
from model import build_model
import keras
from keras.utils.vis_utils import plot_model


def load_config(file):
    ''' Loads a configuration file
    Args:
        file (string): the yaml configuration file's name
    Returns:
        cfg (dict): a dictionary containing the
                         configuration settings
    '''
    try:
        return yaml.load(stream=open(file, 'r'))
    except yaml.YAMLError as ex:
        print(ex)


def dump_config(config):
    ''' Dumps a configuration file
    Args:
        config (dict): a dictionary containing the
                       configuration settings
    '''
    print(yaml.dump(config))


cfg = load_config('Model/config.yaml')
print('Configuration Settings:')
dump_config(cfg)

version = cfg['version']
cfg_model = cfg['model']
model_depth = cfg_model['resnet_depth']
model_name = cfg['name'] + '%dv%d' % (model_depth, version)
print('model name: %s' % model_name)


input_shape = cfg_model['input_shape']

input_tensor = keras.layers.Input(shape=input_shape)
(body, policy_head, value_head) = build_model(input_tensor, cfg_model)

model = keras.models.Model(inputs=[input_tensor], outputs=[
                           policy_head, value_head])

model.summary()

# graphviz has to be installed https://www.graphviz.org/
plot_model(model, to_file='Model/%s.png' % model_name,
           show_shapes=True, show_layer_names=True)
