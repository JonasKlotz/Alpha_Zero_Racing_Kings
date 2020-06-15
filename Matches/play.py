import argparse

from Player import config 
from Azts import player
from Azts import utility
from Matches import match

parser = argparse.ArgumentParser(description="Start command line" \
        "game against ai player.") 
parser.add_argument("-c", "--play_as", \
        type=str, default="white", \
        help="choose color to play with: \"white\" or \"black\" " \
        + "(default: \"white\")")
parser.add_argument("-o", "--opponent", \
        type=str, default="MockingBird", \
        help="choose opponent to play against from available ai players.")
parser.add_argument("-r", "--runs_per_move", \
        type=int, default=100, \
        help="rollouts for the ai engine. for vanilla stockfish " \
        + "behaviour, load a stockfish engine with rollouts set to 1. " \
        + "default is 100")
parser.add_argument("-n", "--name", \
        type=str, default="human player", \
        help="choose human players name.")

args = parser.parse_args()

def cli_play(ai_name, human_name, human_color):

    ai_player = utility.load_player(ai_name) 
    hi_player = player.CLIPlayer(name=human_name) 

    cli_match = None

    if human_color == "white": 
        cli_match = match.Match(player_one=hi_player, \
                player_two=ai_player, \
                runs_per_move=args.runs_per_move, \
                show_game=True) 
    else:
        cli_match = match.Match(player_one=ai_player, \
                player_two=hi_player, \
                runs_per_move=args.runs_per_move, \
                show_game=True) 

    cli_match.simulate()


if __name__ == "__main__":
    cli_play(args.opponent, args.name, args.play_as)


