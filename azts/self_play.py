# pylint: disable=E0401
# pylint: disable=E0602
"""
This module simulates many games of RacingsKings.
"""

import os.path
import time
import pickle

from Player import config

from azts import player
from azts import self_match
from azts import mock_model
from azts import utility
from azts.config import GAMEDIR, \
    RUNS_PER_MOVE, SHOW_GAME

from lib.logger import get_logger
log = get_logger("SelfMatch")

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

    def __init__(self,
                 player_one,
                 player_two,
                 runs_per_move=RUNS_PER_MOVE,
                 game_id="UNNAMED_MATCH",
                 show_game=SHOW_GAME):

        self.players = [player_one, player_two]
        self.runs_per_move = runs_per_move
        self.game_id = game_id
        self.show_game = show_game

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
            log.info(f"\nMATCH {i+1} OF {iterations}:")
            match = self_match.SelfMatch(
                player_one=self.players[switch],
                player_two=self.players[1 - switch],
                runs_per_move=self.runs_per_move,
                show_game=self.show_game)
            match.simulate()
            data = [tuple(j) for j in match.data_collection]

            filepath = utility.get_unused_filepath(
                f"game_{self.game_id}",
                GAMEDIR,
                i)

            pickle.dump(data, open(filepath, "wb"))

            del match
            for i in self.players:
                i.reset()


if __name__ == "__main__":

    player_defs = ("StockingFish", "MockingBird")
    game_id = utility.get_unused_match_handle(*player_defs)
    players = utility.load_players(*player_defs)

    play = SelfPlay(player_one=players[0],
                    player_two=players[1],
                    game_id=game_id,
                    show_game=True)
    play.start(3)
# pylint: enable=E0401
# pylint: enable=E0602
