""" handles configuration parsing from yaml
combines default settings, warning for unknown settings
and type checking
"""
import os
import sys
import yaml

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOTDIR)

from lib.logger import get_logger
log = get_logger("Config")

CONFIGDIR = "Player"


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
        while using default attributes;
        type-assertion for overridden values,
        warning for unknown parameters  
        Args: 
            d (dict): dictionary containing attribute, value to be set
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
                        "Attempted to set member %r (%r) to incorrect type %r." % (
                        member, type(default_value), type(dict_value))
                    setattr(self, member, dict_value)
            else:
                setattr(self, member, default_value)
        for param in d:
            # found parameter in config that does not have set default value
            if not hasattr(default_options, param):
                log.warning("Ignoring unknown parameter %s", param)

    def get_items(self):
        """ Returns a list of tuples (member, value) """
        return [(attr, getattr(self, attr))
                for attr in dir(self)
                if not callable(getattr(self, attr))
                and "__" not in attr]

    def as_dictionary(self):
        """ Returns yaml-style dictionary; recursive for all Options instances """
        d = dict(self.get_items())
        for key, value in d.items():
            if isinstance(value, Options):
                d[key] = value.as_dictionary()
        return d

    def __str__(self, indent=""):
        out = ""
        for member, value in self.get_items():
            out += indent + member + ": "
            if isinstance(value, Options):
                out += "\n" + value.__str__(indent=indent + 4 * " ")
            else:
                out += value.__str__() + "\n"
        return out


class DefaultOptions(Options):

    def __init__(self, d):
        self.load_options_force(d)

    def load_options_force(self, d):
        """ Set member attributes of Options class
        Args:
            d (dict): dictionary containing attribute, value to be set
                      if value is a dictionary itself, another Options class
                      with dictionary values will be added recursively
        """
        for attr, value in d.items():
            if isinstance(value, dict):
                setattr(self, attr, DefaultOptions(value))
            else:
                setattr(self, attr, value)


def search_config_file(name):
    splits = name.split(".yaml")
    if len(splits) == 1:
        name += ".yaml"

    if os.path.exists(name):
        return name

    path = os.path.join(ROOTDIR, name)
    if os.path.exists(path):
        return path

    for _dir in [CONFIGDIR]:
        file = os.path.join(ROOTDIR, _dir, name)

        if os.path.exists(file):
            return file

    raise Exception("Could not find config %s." % name)


class Config(Options):
    """ Config Class """

    def __init__(self, config_file=None):

        default_config_file = search_config_file("default_config")

        default_options = yaml.safe_load(
            stream=open(default_config_file, 'r'))

        if config_file is None:
            log.info("No config provided; using default config.")
            yaml_dict = default_options
        else:
            print(config_file)
            config_file = search_config_file(config_file)
            log.info("Loading Config from %s.", config_file)
            yaml_dict = yaml.safe_load(stream=open(config_file, 'r'))

        default = DefaultOptions(default_options)
        super().__init__(yaml_dict, default)

        self.__yaml_dict = yaml_dict

        # pylint: disable=no-member
        self.model_name = "%s%dv%d" % (self.name,
                                       self.model.resnet_depth,
                                       self.version)
        self.model_uri_format = "models:/%s/{version}" % self.name
        self.model_uri = self.model_uri_format.format(version=self.version)

        # assign directories as members and create if non-existant
        dirs = self.dirs.as_dictionary()
        base = dirs.pop("base")
        for attr, _dir in dirs.items():
            dir_abs = os.path.join(ROOTDIR, base, self.model_name, _dir)
            setattr(self, attr, dir_abs)
            if not os.path.exists(dir_abs):
                log.info("Creating directory %s.", dir_abs)
                os.makedirs(dir_abs)

    def dump(self, file):
        """ Dump yaml to a file;
        Args: file (str): path to file
        """
        with open(file, 'w') as _f:
            yaml.dump(self.as_dictionary(), _f)


if __name__ == "__main__":
    # TEST
    config = Config("SpryGibbon")
    config = Config("Player/SpryGibbon")
    config = Config("SpryGibbon.yaml")
    config = Config()
    config.dump("Player/config.yaml")
