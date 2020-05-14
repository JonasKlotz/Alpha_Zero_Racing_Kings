import os
# import numpy as np
import pickle

from Model.model import AZero
# from Model.mcts_temp import MCTS
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

    def load_train_data(self):
        file = os.path.join(self.config.train_data_dir, 'example.tar')
        with open(file, 'rb') as f:
            self.train_data = pickle.load(f)
        print("training data {} loaded".format(file))

    def save_train_data(self):
        file = os.path.join(self.config.train_data_dir, 'example.tar')
        with open(file, 'wb') as f:
            pickle.dump(self.train_data, f)
        print("training data saved in {}".format(file))

    def run(self):
        pass
        # for _ in range(10):
        #     mcts = MCTS(self.game, self.azero, 1, 100)
        #     self.train_data += self.game_playout(mcts)

    def game_playout(self, mcts):
        pass
        # data = []
        # self.game.reset()
        # while True:
        #     temp = 1
        #     current_player = self.game.player_to_move   # {-1,1}

        #     pi = mcts.getActionProb(temp=temp)
        #     s = self.game.get_state_fen()

        #     action = np.random.choice(self.game.get_move_list_str(), p=pi)
        #     self.game.make_move(action)

        #     r = self.game.get_score(self.game.get_current_player())

        #     r = int((1 - .5) * 2)   # rescale to [-1,1]
        #     # r = r * current_player    # XXX rethink

        #     data.append([s, pi, r])

        #     if r != 0:
        #         return data


if __name__ == "__main__":
    worker = Worker('Model/config.yaml')

    # worker.azero.summary()
    # worker.azero.plot_model()
