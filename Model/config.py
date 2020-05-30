""" handles configuration parsing from yaml
combines default settings, warning for unknown settings
and type checking
"""
import os
import yaml


class Options(object):
    """ Options Class
    able to parse yaml config settings
    while checking for correct data types and unknown settings
    (that aren't contained in the derived class' attributes)
    """

    def __init__(self, file=None, d=None):
        if file is not None:
            self.read_from_yaml(file)
        elif d is not None:
            self.load_options_force(d)

    def load_options_force(self, d):
        """ set member attributes of Options class
        Args: d (dict): dictionary containing attribute, value to be set
                        if value is a dictionary itself, another Options class
                        with dictionary values will be added recursively
        """
        for attr, value in d.items():
            if isinstance(value, dict):
                setattr(self, attr, Options(d=value))
            else:
                setattr(self, attr, value)

    def load_options_safe(self, d):
        """ safely set member attributes of Options class
            while using default attributes (that are already set in instantiated object),
            type-assertion for overridden values
            and warning for unknown parameters (assuming unset members are unused)
        Args: d (dict): dictionary containing attribute, value to be set
                        if value is a dictionary itself, another Options class
                        with dictionary values will be added recursively
        """
        for member in self.get_members():
            if member in d:
                member_value = getattr(self, member)
                dict_value = d[member]
                if isinstance(member_value, Options) and isinstance(dict_value, dict):
                    member_value.load_options_safe(dict_value)
                else:
                    assert isinstance(member_value, type(
                        dict_value)), "attempted to set member %r to incorrect type: %r" % (member, dict_value)
                    setattr(self, member, dict_value)
        for attr in d:
            if not hasattr(self, attr):
                print("!Warning: ignoring unknown parameter %r" % attr)

    def get_members(self):
        """ returns a list of member attributes
        """
        return [attr for attr in dir(self) if not callable(getattr(self, attr))
                and "__" not in attr]

    def read_from_yaml(self, file):
        """ Loads a configuration file
        Args:
            file (str): the yaml configuration file's name
        """
        try:
            self.yaml = yaml.safe_load(stream=open(file, 'r'))
        except yaml.YAMLError as ex:
            print(ex)
            return
        self.load_options_safe(self.yaml)

    def dump_yaml(self, file):
        """ dump yaml to a file
        Args: file (str): path to file
        """
        with open(file, 'w') as _f:
            yaml.dump(self.yaml, _f)

    def __str__(self, indent=""):
        out = ""
        for member in self.get_members():
            value = getattr(self, member)
            out += indent + member + ": "
            if isinstance(value, Options):
                out += "\n" + value.__str__(indent=indent + "    ")
            else:
                out += value.__str__() + "\n"
        return out


class ModelConfig(Options):

    input_shape = [8, 8, 11]
    resnet_depth = 9

    residual_block = Options()
    residual_block.layers = 2
    residual_block.num_filters = 128
    residual_block.filter_size = 3
    residual_block.filter_stride = 1
    residual_block.activation = 'relu'
    residual_block.batch_normalization = True

    policy_head = Options()

    policy_head.residual_layer = Options()
    policy_head.residual_layer.num_filters = 192
    policy_head.residual_layer.filter_size = 3
    policy_head.residual_layer.filter_stride = 1
    policy_head.residual_layer.batch_normalization = True

    policy_head.dense_layer = Options()
    policy_head.dense_layer.num_filters = 64
    policy_head.dense_layer.activation = 'relu'

    value_head = Options()
    value_head.residual_layer = Options()
    value_head.residual_layer.num_filters = 4
    value_head.residual_layer.filter_size = 3
    value_head.residual_layer.filter_stride = 1
    value_head.residual_layer.batch_normalization = True

    value_head.dense_layer = Options()
    value_head.dense_layer.num_filters = 256
    value_head.dense_layer.activation = 'tanh'


class Config(Options):
    """ contains the default config settings
    attributes of interest:
        checkpoint_dir
        self_play_dir
        train_data_dir
    """
    # set the defaults
    name = "AlphaZero"
    config_version = 1
    model_revision = 0

    data_dir = "_Data"

    model = ModelConfig()

    model_name = name + \
        '%dv%dr%d' % (model.resnet_depth, config_version, model_revision)

    checkpoint_dir = os.path.join(data_dir, "model_checkpoints", model_name)
    self_play_dir = os.path.join(data_dir, "selfplay", model_name)
    train_data_dir = os.path.join(data_dir, "training_data")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(self_play_dir):
        os.makedirs(self_play_dir)
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
