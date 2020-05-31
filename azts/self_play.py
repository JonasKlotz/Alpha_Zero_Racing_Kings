# pylint: disable=E0401
# pylint: disable=E0602
"""
This module simulates many games of RacingsKings.
"""

import os.path
import time
import pickle
from azts import self_match
from azts import mock_model
from azts.config import GAMEDIR, \
        RUNS_PER_MOVE, DEFAULT_PLAYER


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
    def __init__(self, \
            player_one=DEFAULT_PLAYER, \
            player_two=DEFAULT_PLAYER, \
            runs_per_move=RUNS_PER_MOVE):

        self.players = [player_one, player_two] 
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
            switch = i % 2 
            print(f"Match {i+1} of {iterations}:")
            print(f"{self.players[switch]['name']} as WHITE " \
                    +f"against {self.players[1-switch]['name']} as BLACK")
            match = self_match.SelfMatch(\
                    player_one=self.players[switch]["azts_settings"], \
                    player_two=self.players[1 - switch]["azts_settings"], \
                    runs_per_move=self.runs_per_move) 
            match.simulate()
            data = [tuple(j) for j in match.data_collection]

            filepath = unused_filename(i)

            pickle.dump(data, open(filepath, "wb"))

            del match



if __name__ == "__main__":

    model = mock_model.MockModel()
    player_one = {"name": "NotoriousWalruss", \
            "azts_settings": {\
                "exploration": 1.1, \
                "heat" : 0.8} \
            }
    player_two = {"name": "WhiteCrane", \
            "azts_settings": {\
                "model": model} \
            }

    play = SelfPlay(player_one=player_one, \
            player_two=player_two)
    play.start(3)
# pylint: enable=E0401
# pylint: enable=E0602
