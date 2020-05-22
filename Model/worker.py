import os
import pickle

from Model.model import AZero
from Interpreter.game import Game
from Model.config import Config


class Worker:

    def __init__(self, config_file=None):
        self.config = Config(config_file)
        self.game = Game()
        self.azero = AZero(self.config)
        self.train_data = []

    def read_config(self, config_file):
        self.config = Config(config_file)

    def load_dummy_data(self):
        file = 'Model/dummy_data.pkl'
        with open(file, 'rb') as f:
            self.train_data = pickle.load(f)
        print("dummy data loaded")

    def load_train_data(self):
        file = os.path.join(self.config.train_data_dir, 'batch.pkl')
        with open(file, 'rb') as f:
            self.train_data = pickle.load(f)
        print("training data loaded from {}".format(file))

    def save_train_data(self):
        file = os.path.join(self.config.train_data_dir, 'batch.pkl')
        with open(file, 'wb') as f:
            pickle.dump(self.train_data, f)
        print("Training data saved to {}".format(file))

    def train(self):
        print("Beginning training")
        self.azero.train(self.train_data)


if __name__ == "__main__":
    worker = Worker('Model/config.yaml')
    worker.load_dummy_data()
    worker.train()

    # worker.azero.summary()
    # worker.azero.plot_model()
