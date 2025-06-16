import glob
import os
import sys
from datetime import datetime
import importlib


class Tee:
    """ Used to redirect and duplicate output """

    def __init__(self, file_name, mode="w"):
        print(f'Tee: redirecting output to {file_name}')
        self.file = open(file_name, mode)
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# decorator to make sure functions clean on ending
def clean_on_end(func):
    def wrapper(*args, **kwargs):
        exp_name = func(*args, **kwargs)
        cleanup(exp_name)
    return wrapper


def cleanup(exp_name):
    print(f'cleanup {exp_name}:')

    today = datetime.now().strftime('%Y%m%d')

    files = []
    for ext in ['.dill', '.keras']:
        files.extend(
            glob.glob(os.path.join(f'experiments/{exp_name}', "**", f"*{ext}"),
                      recursive=True)
        )

    for dfile in files:
        if today in dfile:
            print(f'deleting {dfile}')
            os.remove(dfile)


def load_config(obj, config_name):
    # Load a config that lives in the <configs> dir: config_file =
    # <configs>/<config_name>.py. Overwrite class members and
    # create new ones according to what is present in
    # config_file. Exclude "__" members and functions.

    config_file = f'configs.{config_name}'
    print(f'Load config: {config_file}')

    try:
        config = importlib.import_module(config_file)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(config_file)

    importlib.reload(config)

    module_vars = vars(config)

    # load variables
    config_vars = {
        key: value for key, value in module_vars.items()
        if not key.startswith("__") and not callable(value)
    }

    for (key, value) in config_vars.items():
        setattr(obj, key, value)
