'''
this instantiates an analysis match which
tracks metrics on the mlflow server
'''
import time
import argparse
import mlflow
import pickle
import os 
from lib.logger import get_logger

from Azts import utility 
from Azts.config import TO_STRING, WHITE, BLACK
from Matches import match 

log = get_logger("AnalysisMatch")

class AnalysisMatch(match.Match):

    match_moves = []
    gamestats = []

    def simulate(self):
        '''
        simulate a game. this starts a
        loop of taking turns and making
        moves between the players while
        storing each game position and
        corresponding move distributions
        in data collection. loop ends with
        end of match.

        overloading function to track
        lots of statistics.

        :return int: state in which game
        ended according to enum type
        defined in config: running, white
        wins, black wins, draw, draw by
        stale mate, draw by repetition,
        draw by two wins
        '''
        moves = 1
        time1 = time.time()
        log.info(f"\nWHITE: {self.players[0].name}\n"
              + f"BLACK: {self.players[1].name}\n")

        # game loop:
        while True:
            # check break condition:
            if self.game.is_ended():
                break
            # select players
            select = 0 if self.game.get_current_player() else 1
            active_player = self.players[select]
            other_player = self.players[1 - select]
            # handle all moves
            move = active_player.make_move()
            other_player.receive_move(move)
            self.game.make_move(move)
            self.match_moves.append(move)

            # statistics:
            # only increment after black move
            moves += select
            self._show_game()

            if moves % self.report_cycle == 0 and select == 0:
                time1 = self._report(time1, moves)


            if moves % self.report_cycle == 0 and \
                    active_player.color == self.track_player:
                stats = active_player.get_stats()
                stats = {} if stats == None else stats 
                color = 1 if active_player.color == WHITE else -1
                stats["player_color"] = color
                self.gamestats.append(stats)


        result = self.game.board.result()
        state = self.game.get_game_state()
        log.info(f"game ended after {moves} "
              + f"moves with {result} ({TO_STRING[state]}).")
        score = self.training_payoffs[state]

        # write results to mlserver
        if self.track_player in [WHITE, BLACK]:
            with mlflow.start_run():
                log.info("writing to mlflow server: score ...")
                mlflow.log_metric("score", score) 

                log.info("writing to mlflow server: match moves ...")
                utility.write_artifact_to_server(self.match_moves, "match_moves") 

                log.info("writing to mlflow server: metrics ...")
                for i, j in enumerate(self.gamestats):
                    move = (i + 1) * self.report_cycle
                    utility.unpack_metrics(dictionary=j, idx=move)

        for i in self.players:
            i.stop() 

        return state 

    def track_metrics(self):
        result = self.simulate()
        score = self.training_payoffs[result]

        with mlflow.start_run():
            mlflow.log_metric("score", score)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Track metrics in a match " \
            + "between two selected ai players.")
    parser.add_argument("--player_one",
            type=str, default="AltruisticOlm", \
            help="A player in the analysis match.")
    parser.add_argument("--player_two",
            type=str, default="AltruisticOlm", \
            help="Other player in the analysis match.") 
    parser.add_argument("-r", "--rollouts_per_move",
            type=int, default=100, \
            help="Simulation runs for each move.")
    parser.add_argument("-t", "--tracked_player_color", \
            type=str, default="white", \
            help="Select for which player statistics are " \
            + "being tracked: black or white. Default: white.")
    parser.add_argument("-c", "--report_cycle", \
            type=int, default=10, \
            help="After how many full turns statistics are being logged. " \
            + "Default is 10")
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
    start_args["report_cycle"] = args.report_cycle
    start_args["track_player"] = args.tracked_player_color

    analysis_match = AnalysisMatch(**start_args)
    analysis_match.simulate()
