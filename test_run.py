import time
import pickle

from Model.model import AZero
from Model.config import Config


def timing(func):
    """ wrapper for timing functions
    usage:
    @timing
    def time_test():
        myfunc()
    time_test()
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = func(*args, **kwargs)
        total = time.time() - time1
        hours = int(total / 3600)
        minutes = int((total % 3600) / 60)
        seconds = int((total % 60))
        if hours is not 0:
            print('{:s}() took {}h {}m {}s'.format(
                func.__name__, hours, minutes, seconds))
        elif minutes is not 0:
            print('{:s}() took {}m {}s'.format(
                func.__name__, minutes, seconds))
        elif seconds is not 0:
            print('{:s}() took {:.2f}s'.format(func.__name__, total))
        else:
            print('{:s}() took {:.2f}ms'.format(func.__name__, total * 1000))
        return ret
    return wrap


if __name__ == "__main__":
    print("===== Executing Test Run =====")
    config = Config('Model/config.yaml')
    azero = AZero(config)
    FILE = '_Data/training_data/dataset_685_games.pkl'
    #FILE = '_Data/training_data/game_0000.pkl'
    with open(FILE, 'rb') as f:
        train_data = pickle.load(f)

    @timing
    def time_test():
        azero.train(train_data)

    time_test()
