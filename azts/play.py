from Player import config

from azts import player
from azts import utility
from azts import self_match



ai_player = utility.load_player("SpryGibbon", mock=True)

hi_player = player.CLIPlayer("FrankSquid")


match = self_match.SelfMatch(ai_player, hi_player, 20, True)

match.simulate()


