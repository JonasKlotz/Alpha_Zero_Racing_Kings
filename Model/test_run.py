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
        t = time.time() - time1
        h = int(t / 3600)
        m = int((t % 3600) / 60)
        s = int((t % 60))
        if h is not 0:
            print('{:s}() took {}h {}m {}s'.format(func.__name__, h, m, s))
        elif m is not 0:
            print('{:s}() took {}m {}s'.format(func.__name__, m, s))
        elif s is not 0:
            print('{:s}() took {:.2f}s'.format(func.__name__, t))
        else:
            print('{:s}() took {:.2f}ms'.format(func.__name__, t*1000))
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
