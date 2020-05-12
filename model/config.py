import yaml


def load_yaml(file):
    ''' Loads a configuration file
    Args:
        file (string): the yaml configuration file's name
    '''
    try:
        return yaml.load(stream=open(file, 'r'))
    except yaml.YAMLError as ex:
        print(ex)


class Options(object):
    '''Options Class
    able to parse yaml config settings
    while checking for correct data types and unknown settings
    '''

    def __init__(self, file=None, d=None):
        if file is not None:
            self.read_from_yaml(file)
        elif d is not None:
            self.load_options_force(d)

    def load_options_force(self, d):
        for attr, value in d.items():
            if type(value) is dict:
                setattr(self, attr, Options(d=value))
            else:
                setattr(self, attr, value)

    def load_options_safe(self, d):
        for member in self.get_members():
            if member in d:
                member_value = getattr(self, member)
                dict_value = d[member]
                if type(member_value) is Options and type(dict_value) is dict:
                    member_value.load_options_safe(dict_value)
                else:
                    assert type(member_value) == type(
                        dict_value), "attempted to set member %r to incorrect type: %r" % (member, dict_value)
                    setattr(self, member, dict_value)
        for attr in d:
            if not hasattr(self, attr):
                print("!Warning: ignoring unknown parameter %r" % attr)

    def get_members(self):
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and "__" not in attr]

    def read_from_yaml(self, file):
        self.load_options_safe(load_yaml(file))

    def __str__(self, indent=""):
        out = ""
        for member in self.get_members():
            value = getattr(self, member)
            out += indent + member + ": "
            if type(value) == Options:
                out += "\n" + value.__str__(indent=indent + "    ")
            else:
                out += value.__str__() + "\n"
        return out


class Config(Options):
    # set the defaults
    name = "AlphaZero"
    version = 1

    model = Options()
    model.input_shape = (8, 8, 11)
    model.resnet_depth = 9

    model_name = name + '%dv%d' % (model.resnet_depth, version)

    model.residual_block = Options()
    model.residual_block.layers = 2
    model.residual_block.num_filters = 128
    model.residual_block.filter_size = 3
    model.residual_block.filter_stride = 1
    model.residual_block.activation = 'relu'
    model.residual_block.batch_normalization = True

    model.policy_head = Options()

    model.policy_head.residual_layer = Options()
    model.policy_head.residual_layer.num_filters = 192
    model.policy_head.residual_layer.filter_size = 3
    model.policy_head.residual_layer.filter_stride = 1
    model.policy_head.residual_layer.batch_normalization = True

    model.policy_head.dense_layer = Options()
    model.policy_head.dense_layer.num_filters = 64
    model.policy_head.dense_layer.activation = 'relu'

    model.value_head = Options()
    model.value_head.residual_layer = Options()
    model.value_head.residual_layer.num_filters = 4
    model.value_head.residual_layer.filter_size = 3
    model.value_head.residual_layer.filter_stride = 1
    model.value_head.residual_layer.batch_normalization = True

    model.value_head.dense_layer = Options()
    model.value_head.dense_layer.num_filters = 256
    model.value_head.dense_layer.activation = 'tanh'
