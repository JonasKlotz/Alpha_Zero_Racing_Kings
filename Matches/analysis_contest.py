# pylint: disable=E0401
# pylint: disable=E0602
"""
This module simulates many games of RacingsKings.
"""

import os.path
import time
import argparse
import pickle 
import mlflow
from lib.logger import get_logger

from Player import config 
from Azts import player
from Azts import mock_model
from Azts import utility
from Azts.config import GAMEDIR, \
    ROLLOUTS_PER_MOVE, SHOW_GAME, \
    WHITE, BLACK, \
    WHITE_WINS, BLACK_WINS, \
    DRAW_BY_REP, DRAW_BY_TWO_WINS, \
    DRAW_BY_STALE_MATE
from Matches import match
from Matches import contest

log = get_logger("AnalysisContest")

WINS = "_wins"
LOSSES = "_losses"
DRAWS = "_draws"

class AnalysisContest(contest.Contest):
    
    gamestats = {}


    def start(self, num_of_matches=10):
        '''
        start a series of matches. match data
        for each match is written to a separate
        file in the games folder as defined in
        config.
        :param int num_of_matches: number of matches
        to be simulated
        '''
        for i in self.players:
            self.gamestats[i.name] = {}
            for j in [WHITE, BLACK]:
                self.gamestats[i.name][j] = {}
                for k in [WHITE_WINS, BLACK_WINS, \
                        DRAW_BY_REP, DRAW_BY_TWO_WINS, \
                        DRAW_BY_STALE_MATE]:
                    self.gamestats[i.name][j][k] = 0
            for j in [WINS, LOSSES, DRAWS]:
                self.gamestats[i.name][j] = 0


        for i in range(num_of_matches):
            switch = i % 2
            log.info(f"\nMATCH {i+1} OF {num_of_matches}:")
            
            player_one = self.players[switch]
            player_two = self.players[1 - switch]


            contest_match = match.Match(
                player_one=player_one,
                player_two=player_two,
                rollouts_per_move=self.rollouts_per_move,
                show_game=self.show_game)

            result = contest_match.simulate()

            for j, k in zip([player_one, player_two], [WHITE, BLACK]):
                self.gamestats[j.name][k][result] += 1 


            if result == WHITE_WINS:
                self.gamestats[player_one.name][WINS] += 1
                self.gamestats[player_two.name][LOSSES] += 1
            elif result == BLACK_WINS:
                self.gamestats[player_one.name][LOSSES] += 1
                self.gamestats[player_two.name][WINS] += 1
            else:
                self.gamestats[player_one.name][DRAWS] += 1
                self.gamestats[player_two.name][DRAWS] += 1 

            del contest_match
            for j in self.players:
                j.reset()

        with mlflow.start_run():
            self._unpack_metrics(self.gamestats) 


    def _unpack_metrics(self, dictionary, prefix=""):
        '''
        call this function within a mlflow run
        environment to unpack statistic dictionaries
        from the model and track them in mlflow
        :param int moves: current move in game 
        '''
        for i in dictionary.keys():
            j = dictionary[i]
            new_prefix = f"{i}" if prefix is "" else f"{prefix}-{i}"
            if isinstance(j, dict):
                self._unpack_metrics(j, new_prefix)
            elif isinstance(j, float) or isinstance(j, int):
                mlflow.log_metric(new_prefix, j)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Track outcomes " \
            + "of a contest between ai players.")
    parser.add_argument("--player_one",
            type=str, default="AltruisticOlm", \
            help="A player in the analysis contest.")
    parser.add_argument("--player_two",
            type=str, default="AltruisticOlm", \
            help="Other player in the analysis contest.") 
    parser.add_argument("-r", "--rollouts_per_move",
            type=int, default=100, \
            help="Simulation runs for each move.")
    parser.add_argument("-n", "--number_of_games_in_contest",
            type=int, default=10, \
            help="number of games to be played in this contest. Default 10.")
    parser.add_argument("--show_game",
            type=int, default=0, \
            help="Show game - 1 for yes, 0 for no. 0 is default.")

    args = parser.parse_args()

    start_args = {}
    for i, j in zip(["player_one", "player_two"], \
            [args.player_one, args.player_two]):
        loaded_player = utility.load_player(j)
        start_args[i] = loaded_player

    start_args["rollouts_per_move"] = args.rollouts_per_move
    start_args["show_game"] = bool(args.show_game)

    analysis_contest = AnalysisContest(**start_args)
    analysis_contest.start(args.number_of_games_in_contest)
