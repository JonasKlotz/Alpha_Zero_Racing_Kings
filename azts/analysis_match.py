import time
import argparse
import mlflow

from azts import self_match
from azts import utility 
from azts.config import TO_STRING

from lib.logger import get_logger
log = get_logger("AnalysisMatch")

REPORT_CYCLE = 10

class AnalysisMatch(self_match.SelfMatch):

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
        with mlflow.start_run():
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
                # collect data
                self.data_collection.append(active_player.dump_data())

                # statistics:
                # only increment after black move
                moves += select
                self._show_game()
                if moves % REPORT_CYCLE == 0 and select:
                    stats = active_player.get_stats()
                    # in stats: settings, tree, move distribution
                    #move_dist_stats = \
                            #utility.filter_non_numbers_from_dict(\
                            #stats["move distribution"])
                    self._unpack_metrics(stats, moves)
                    mlflow.log_metric("move", moves, moves)
                    time1 = self._report(time1, moves)

            result = self.game.board.result()
            state = self.game.get_game_state()
            log.info(f"game ended after {moves} "
                  + f"moves with {result} ({TO_STRING[state]}).")
            score = self.training_payoffs[state]
            mlflow.log_metric("score", score)

        for i in self.data_collection:
            i[2] = score

        return state


    def _unpack_metrics(self, dictionary, moves):
        '''
        call this function within a mlflow run
        environment to unpack statistic dictionaries
        from the model and track them in mlflow
        :param int moves: current move in game 
        '''
        for i in dictionary.keys():
            j = dictionary[i]
            if isinstance(j, dict):
                self._unpack_metrics(j, moves)
            elif isinstance(j, float) or isinstance(j, int):
                mlflow.log_metric(i, j, moves)


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
    parser.add_argument("-r", "--runs_per_move",
            type=int, default=100, \
            help="Simulation runs for each move.")
    parser.add_argument("--show_game",
            type=int, default=0, \
            help="Show game - 1 for yes, 0 for no. 0 is default.")

    args = parser.parse_args()

    start_args = {}
    for i, j in zip(["player_one", "player_two"], \
            [args.player_one, args.player_two]):
        loaded_player = utility.load_player(j)
        start_args[i] = loaded_player

    start_args["runs_per_move"] = args.runs_per_move
    start_args["show_game"] = bool(args.show_game)

    match = AnalysisMatch(**start_args)
    match.simulate()
