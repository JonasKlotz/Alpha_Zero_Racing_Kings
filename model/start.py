#!/usr/bin/env python3

from Model.model import AZero

# from mcts_temp import MCTS
# from Interpreter.game import Game

# azero = AZero('Model/config.yaml')
# azero.summary()
# graphviz (not a python package) has to be installed https://www.graphviz.org/
# azero.plot_model()


class Worker:

    def run(self):
        azero = AZero('Model/config.yaml')
        # azero.train()
