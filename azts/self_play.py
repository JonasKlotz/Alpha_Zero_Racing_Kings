# pylint: disable=E0401
# pylint: disable=E0602
"""
This module simulates many games of RacingsKings.
"""

import os.path
import time
import pickle
from azts import self_match
from azts.config import *


def unused_filename(i=0):
    '''
    function to find the lowest unused
    filename within games folder according
    to naming scheme "game_0000.pkl"
    '''
    filenumber = i

    filenumberstring = str(filenumber).zfill(4)
    filename = f"game_{filenumberstring}.pkl"
    filepath = os.path.join(GAMEDIR, filename)
    while os.path.isfile(filepath):
        filenumber += 1
        filenumberstring = str(filenumber).zfill(4)
        filename = f"game_{filenumberstring}.pkl"
        filepath = os.path.join(GAMEDIR, filename)

    return filepath


class SelfPlay():
    '''
    selfplay is initialized with the number of
    rollouts that the matching ai player are
    using per move.
    the number of game simulations is determined
    by the parameter in function start() which
    actually starts the series of matches.
    After each match, the match data is written
    to a separate file which facilitates
    parallelisation of creating data for many
    matches.
    '''
    def __init__(self, runs_per_move=RUNS_PER_MOVE):
        self.match = self_match.SelfMatch(runs_per_move)
        self.runs_per_move = runs_per_move

    def start(self, iterations=10):
        '''
        start a series of matches. match data
        for each match is written to a separate
        file in the games folder as defined in
        config.
        :param int iterations: number of matches
        to be simulated
        '''
        for i in range(iterations):
            self.match.simulate()
            data = [tuple(j) for j in self.match.data_collection]

            filepath = unused_filename(i)

            pickle.dump(data, open(filepath, "wb"))

            del self.match
            self.match = self_match.SelfMatch(self.runs_per_move) 


if __name__ == "__main__":
    play = SelfPlay()
    play.start(3)
# pylint: enable=E0401
# pylint: enable=E0602
