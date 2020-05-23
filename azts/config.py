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


# Data types for size of
# np.arrays
# np.float16:
# no overflow in -65500 .. 65500
MOVE_DTYPE = np.uint8
POS_DTYPE = np.uint8
EDGE_DTYPE = np.float16
IDX_DTYPE = np.uint16
