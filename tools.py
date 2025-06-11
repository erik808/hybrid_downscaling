import glob
import os
import sys
from datetime import datetime


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
