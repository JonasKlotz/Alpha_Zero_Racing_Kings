import argparse
import mlflow

from azts import self_match
from azts import utility 

class AnalysisMatch(self_match.SelfMatch):

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
    match.track_metrics()
