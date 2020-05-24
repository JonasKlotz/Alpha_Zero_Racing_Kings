"""
This module simulates games of RacingsKings.
"""

import os.path
import time
import pickle
from azts import self_match
from azts.config import *

REPORT_CYCLE = 25 

def unused_filename(i = 0):
    filenumber = i

    filenumberstring = str(filenumber).zfill(4)
    filename = f"game_{filenumberstring}.pkl"
    filepath = os.path.join(GAMEDIR, filename)
    while os.path.isfile(filepath):
        filenumber += 1
        filenumberstring = str(filenumber).zfill(4)
        filename = f"game_{filenumberstring}.pkl"
        filepath = os.path.join(GAMEDIR, filename)

    return filepath


class SelfPlay():
    def __init__(self):
        self.match = self_match.SelfMatch(RUNS_PER_MOVE)

    def start(self, iterations=10):
        for i in range(iterations):
            self.match.simulate()
            data = [tuple(j) for j in self.match.data_collection]

            filepath = unused_filename(i)

            pickle.dump(data, open(filepath, "wb"))

            del self.match
            self.match = self_match.SelfMatch(RUNS_PER_MOVE) 


if __name__ == "__main__":
    play = SelfPlay()
    play.start(3)
