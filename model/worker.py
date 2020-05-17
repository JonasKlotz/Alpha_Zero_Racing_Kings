import os
import pickle

from Model.model import AZero
# from Interpreter.game import Game
from Model.config import Config


class Worker:

    def __init__(self, config_file=None):
        self.config = Config(config_file)
        # self.game = Game()
        self.azero = AZero(self.config)
        # self.train_data = []

    def train(self, train_data):
        self.azero.train(train_data)

    def load_train_data(self):
        file = os.path.join(self.config.train_data_dir, 'batch.pkl')
        with open(file, 'rb') as f:
            return pickle.load(f)

    def save_self_play(self, data):
        file = os.path.join(self.config.self_play_dir, 'batch.pkl')
        with open(file, 'wb') as f:
            pickle.dump(data, f)
        print("Self play data saved to {}".format(file))
