""" handles configuration parsing from yaml
combines default settings, warning for unknown settings
and type checking
"""
import os
import yaml

from lib.logger import get_logger
log = get_logger("config")


class Options(object):
    """ Options Class
    able to parse yaml config settings
    while checking for correct data types and unknown settings
    (that aren't contained in the derived class' attributes)
    """

    def __init__(self, d, default):
        self.load_options_safe(d, default)

    def load_options_safe(self, d, default_options):
        """ safely set member attributes of Options class
            while using default attributes (that are already set in instantiated object),
            type-assertion for overridden values
            and warning for unknown parameters (assuming unset members are unused)
        Args: d (dict): dictionary containing attribute, value to be set
                        if value is a dictionary itself, another Options class
                        with dictionary values will be added recursively
        """
        for member, default_value in default_options.get_items():
            if member in d:  # default parameter will be overwritten
                dict_value = d[member]
                if isinstance(default_value, DefaultOptions) and isinstance(dict_value, dict):
                    opt = Options(dict_value, default_value)
                    setattr(self, member, opt)
                else:
                    assert isinstance(default_value, type(dict_value)), \
                        "attempted to set member %r (%r) to incorrect type %r" % (
                        member, type(default_value), type(dict_value))
                    setattr(self, member, dict_value)
            else:
                setattr(self, member, default_value)
        for param in d:
            # found parameter in config that does not have set default value
            if not hasattr(default_options, param):
                log.warning("ignoring unknown parameter " + param)

    def get_items(self):
        """ returns a list of member attributes
        """
        return [(attr, getattr(self, attr)) for attr in dir(self) if not callable(getattr(self, attr))
                and "__" not in attr]

    def get_dictionary(self):
        return dict(self.get_items())

    def __str__(self, indent=""):
        out = ""
        for member, value in self.get_items():
            out += indent + member + ": "
            if isinstance(value, Options):
                out += "\n" + value.__str__(indent=indent + "    ")
            else:
                out += value.__str__() + "\n"
        return out


class DefaultOptions(Options):

    def __init__(self, d):
        self.load_options_force(d)

    def load_options_force(self, d):
        """ set member attributes of Options class
        Args: d (dict): dictionary containing attribute, value to be set
                        if value is a dictionary itself, another Options class
                        with dictionary values will be added recursively
        """
        for attr, value in d.items():
            if isinstance(value, dict):
                setattr(self, attr, DefaultOptions(value))
            else:
                setattr(self, attr, value)


class Config(Options):
    """ contains the default config settings
    attributes of interest:
        checkpoint_dir
        self_play_dir
        train_data_dir
    """
    DEFAULT_CONFIG_FILE = "Player/default_config.yaml"

    def __init__(self, config_file):
        yaml_default = yaml.safe_load(
            stream=open(self.DEFAULT_CONFIG_FILE, 'r'))
        default = DefaultOptions(yaml_default)
        yaml_dict = yaml.safe_load(stream=open(config_file, 'r'))
        self.__yaml_dict = yaml_dict
        super().__init__(yaml_dict, default)

        # pylint: disable=no-member
        self.model_name = self.name + \
            '%dv%dr%d' % (self.model.resnet_depth,
                          self.config_version,
                          self.model_revision)

        # set directories
        self.checkpoint_dir = os.path.join(
            self.data_dir, "model_checkpoints", self.model_name)
        self.self_play_dir = os.path.join(
            self.data_dir, "selfplay", self.model_name)
        self.train_data_dir = os.path.join(self.data_dir, "training_data")

        # create directories if non-existant
        for _dir in [self.checkpoint_dir, self.self_play_dir, self.train_data_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

    def dump_yaml(self, file):
        """ dump yaml to a file
        Args: file (str): path to file
        """
        with open(file, 'w') as _f:
            yaml.dump(self.__yaml_dict, _f)


if __name__ == "__main__":
    # TEST
    config = Config("Player/config.yaml")
    print(config)
