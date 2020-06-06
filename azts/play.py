import argparse

from Player import config

from azts import player
from azts import utility
from azts import self_match

parser = argparse.ArgumentParser(description="Start command line" \
        "game against ai player.") 
parser.add_argument("-c", "--play_as", \
        type=str, default="white", \
        help="choose color to play with: \"white\" or \"black\" " \
        + "(default: \"white\")")
parser.add_argument("-o", "--opponent", \
        type=str, default="MockingBird", \
        help="choose opponent to play against from available ai players.")
parser.add_argument("-n", "--name", \
        type=str, default="human player", \
        help="choose human players name.")

args = parser.parse_args()

def cli_play(ai_name, human_name, human_color):

    ai_player = utility.load_player(ai_name) 
    hi_player = player.CLIPlayer(name=human_name) 

    match = None

    if human_color == "white": 
        match = self_match.SelfMatch(hi_player, ai_player, 20, True) 
    else:
        match = self_match.SelfMatch(ai_player, hi_player, 20, True) 

    match.simulate()


if __name__ == "__main__":
    cli_play(args.opponent, args.name, args.play_as)


