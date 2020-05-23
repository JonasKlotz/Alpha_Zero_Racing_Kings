"""
Configuration for AlphaZeroTreeSearch modules (azts).
More local configuration parameters can be found in azts.py
"""

import numpy as np

# Paths
GAMEDIR = "games"  # directory for self_play to store datasets in.
SELFPLAY = True
EXPLORATION = 0.1
AMPLIFY_RESULT = 100

# Misc
RUNS_PER_MOVE = 100  # Sets the number of azts runs
SHOW_GAME = False  # If True boards will be shown in self_play

# Enum Types representing
# player colors
WHITE = 1
BLACK = -1

# Enum Types representing
# all possible game states
RUNNING = 0
WHITE_WINS = 1
BLACK_WINS = 2
DRAW = 3
DRAW_BY_REP = 4
DRAW_BY_STALE_MATE = 5
DRAW_BY_TWO_WINS = 6
NUM_OF_OUTCOMES = 7

PAYOFFS = {WHITE_WINS: 1, \
        BLACK_WINS: -1, \
        DRAW: 0, \
        DRAW_BY_REP: 0, \
        DRAW_BY_STALE_MATE: 0, \
        DRAW_BY_TWO_WINS: 0}

TO_STRING = {0: "undefined", \
        WHITE_WINS: "white won", \
        BLACK_WINS: "black won", \
        DRAW: "draw", \
        DRAW_BY_REP: "draw by repetition", \
        DRAW_BY_STALE_MATE: "draw by stale mate", \
        DRAW_BY_TWO_WINS: "draw by simultaneous finish"}

# Data types for size of
# np.arrays
# np.float16:
# no overflow in -65500 .. 65500
MOVE_DTYPE = np.uint8
POS_DTYPE = np.uint8
EDGE_DTYPE = np.float16
IDX_DTYPE = np.uint16
