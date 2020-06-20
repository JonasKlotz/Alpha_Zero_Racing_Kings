# pylint: disable=E0401
# pylint: disable=E0602
"""
This module simulates many games of RacingsKings.
"""

import os.path
import time
import pickle 
from lib.logger import get_logger

from Player import config 
from Azts import player
from Azts import mock_model
from Azts import utility
from Azts.config import GAMEDIR, \
    ROLLOUTS_PER_MOVE, SHOW_GAME
from Matches import match

log = get_logger("Contest")

class Contest():
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
                 rollouts_per_move=ROLLOUTS_PER_MOVE,
                 game_id="UNNAMED_MATCH",
                 show_game=SHOW_GAME):

        self.players = [player_one, player_two]
        self.rollouts_per_move = rollouts_per_move
        self.game_id = game_id
        self.show_game = show_game

    def start(self, num_of_matches=3):
        '''
        start a series of matches. match data
        for each match is written to a separate
        file in the games folder as defined in
        config.
        :param int num_of_matches: number of matches
        to be simulated
        '''
        for i in range(num_of_matches):
            switch = i % 2
            log.info(f"\nMATCH {i+1} OF {num_of_matches}:")
            contest_match = match.Match(
                player_one=self.players[switch],
                player_two=self.players[1 - switch],
                rollouts_per_move=self.rollouts_per_move,
                show_game=self.show_game)

            contest_match.simulate()
            data = [tuple(j) for j in contest_match.data_collection]

            filepath = utility.get_unused_filepath(
                f"game_{self.game_id}",
                GAMEDIR,
                i)
            print(filepath)
            pickle.dump(data, open(filepath, "wb"))

            del contest_match
            for i in self.players:
                i.reset()


if __name__ == "__main__":

    player_defs = ("StockingFish", "MockingBird")
    game_id = utility.get_unused_match_handle(*player_defs)
    players = utility.load_players(*player_defs)

    play = Contest(player_one=players[0],
                    player_two=players[1],
                    game_id=game_id,
                    show_game=True)
    play.start(num_of_matches=3)
# pylint: enable=E0401
# pylint: enable=E0602
