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
from azts.config import GAMEDIR, \
        RUNS_PER_MOVE, DEFAULT_PLAYER, \
        SHOW_GAME


def unused_filename(names, game_id, i=0):
    '''
    function to find the lowest unused
    filename within games folder according
    to naming scheme "game_[NAME]-[NAME]_0000.pkl"
    where the names are sorted alphabetically
    '''
    filenumber = i

    names.sort()
    namestring = names[0] + "-" + names[1]
    game_name = f"game_{game_id}_{namestring}"

    filenumberstring = str(filenumber).zfill(4)
    filename = f"{game_name}_{filenumberstring}.pkl"
    filepath = os.path.join(GAMEDIR, filename)
    while os.path.isfile(filepath):
        filenumber += 1
        filenumberstring = str(filenumber).zfill(4)
        filename = f"{game_name}_{filenumberstring}.pkl"
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
            player_one, \
            player_two, \
            runs_per_move=RUNS_PER_MOVE, \
            game_id="game-id", \
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
            print(f"\nMATCH {i+1} OF {iterations}:")
            match = self_match.SelfMatch(\
                    player_one=self.players[switch], \
                    player_two=self.players[1 - switch], \
                    runs_per_move=self.runs_per_move, \
                    show_game=self.show_game)
            match.simulate()
            data = [tuple(j) for j in match.data_collection]

            filepath = unused_filename([\
                    self.players[0].name, \
                    self.players[1].name], \
                    self.game_id, \
                    i)

            pickle.dump(data, open(filepath, "wb"))

            del match
            for i in self.players:
                i.reset()



if __name__ == "__main__":

    model = mock_model.MockModel()

    conf_one = config.Config("Player/default_config.yaml") 
    player_one = player.Player(name=conf_one.name, \
            model=model, \
            **(conf_one.player.as_dictionary()))

    conf_two = config.Config("Player/SpryGibbon.yaml")
    player_two = player.Player(name=conf_two.name, \
            model=model, \
            **(conf_two.player.as_dictionary()))

    play = SelfPlay(player_one=player_one, \
            player_two=player_two)
    play.start(3)
# pylint: enable=E0401
# pylint: enable=E0602
