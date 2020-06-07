"""
Configuration for AlphaZeroTreeSearch modules (azts).
More local configuration parameters can be found in azts.py
"""

import numpy as np
import sys
import os.path
import os

from lib.logger import get_logger
log = get_logger("azts.config")

from azts import mock_model


def find_rootdir():
    rootdir = os.path.split(
        os.path.dirname(
            os.path.join(os.getcwd(), __file__)))[0]

    if rootdir is None:
        raise Exception("Could not find Root Directory!\n"
                        + "Please set PYTHONPATH variable to project:\n"
                        + f"Navigate to project root folder and type:\n"
                        + "export PYTHONPATH=`pwd`")

    if rootdir not in sys.path:
        sys.path.append(rootdir)

    return rootdir


ROOTDIR = find_rootdir()
#log.info(f"Rootdir is {ROOTDIR}")

# Paths
# Folder to store self-game files,
# auxiliary files (pictures etc.)
# and complete datasets
GAMEFOLDER = "games"
RESOURCESFOLDER = "resources"
DATASETFOLDER = "datasets"
PLAYERFOLDER = "Player"

GAMEDIR = os.path.join(ROOTDIR, GAMEFOLDER)
RESOURCESDIR = os.path.join(ROOTDIR, RESOURCESFOLDER)
DATASETDIR = os.path.join(ROOTDIR, DATASETFOLDER)
PLAYERDIR = os.path.join(ROOTDIR, PLAYERFOLDER)

for i in [GAMEDIR, RESOURCESDIR, DATASETDIR]:
    if not os.path.exists(i):
        log.info(f"Could not find {i} -- making dir {i}")
        os.makedirs(i)

EXPLORATION = 0.1
HEAT = 1

# Misc
RUNS_PER_MOVE = 1  # Sets the number of azts runs
SHOW_GAME = True  # If True boards will be shown in self_play

# Enum Types representing
# player colors
WHITE = "white"
BLACK = "black"

# Enum Types representing
# all possible game states
RUNNING = "running"
WHITE_WINS = "white_wins"
BLACK_WINS = "black_wins"
DRAW = "draw"
DRAW_BY_REP = "draw_by_rep"
DRAW_BY_STALE_MATE = "draw_by_stale_mate"
DRAW_BY_TWO_WINS = "draw_by_two_wins"

TRAINING_PAYOFFS = {WHITE_WINS: 1,
                    BLACK_WINS: -1,
                    DRAW: 0,
                    DRAW_BY_REP: 0,
                    DRAW_BY_STALE_MATE: 0,
                    DRAW_BY_TWO_WINS: 0}

ROLLOUT_PAYOFFS = {WHITE: {WHITE_WINS: 1,
                           BLACK_WINS: -1,
                           DRAW: 0,
                           DRAW_BY_REP: 0,
                           DRAW_BY_STALE_MATE: 0,
                           DRAW_BY_TWO_WINS: 0},
                   BLACK: {WHITE_WINS: -1,
                           BLACK_WINS: 1,
                           DRAW: 0,
                           DRAW_BY_REP: 0,
                           DRAW_BY_STALE_MATE: 0,
                           DRAW_BY_TWO_WINS: 0}}

TO_STRING = {0: "undefined",
             WHITE_WINS: "white won",
             BLACK_WINS: "black won",
             DRAW: "draw",
             DRAW_BY_REP: "draw by repetition",
             DRAW_BY_STALE_MATE: "draw by stale mate",
             DRAW_BY_TWO_WINS: "draw by simultaneous finish"}

# Data types for size of
# np.arrays
# np.float16:
# no overflow in -65500 .. 65500
MOVE_DTYPE = np.uint8
POS_DTYPE = np.uint8
EDGE_DTYPE = np.float16
IDX_DTYPE = np.uint16
