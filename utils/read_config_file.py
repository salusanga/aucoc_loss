import os
import yaml
from yaml.loader import FullLoader


def read_config_file(path, config_file, args):
    """Function to read config file and add parameters to argparse"""
    with open(os.path.join(path, config_file + ".yaml")) as f:
        args_file_loaded = yaml.load(f, Loader=FullLoader)
    print("Loading arguments from file:", config_file)
    args.update(args_file_loaded)
    return args
