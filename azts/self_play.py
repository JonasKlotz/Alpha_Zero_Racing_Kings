"""
This module simulates games of RacingsKings.
"""

import os.path
import time
import pickle
import azts
from Interpreter import game
import state_machine as sm
import self_match
import screen
from config import *

REPORT_CYCLE = 25



class SelfPlay():
    def __init__(self):
        self.match = self_match.SelfMatch()

    def start(self, iterations=100):
        for i in range(iterations):
            self.match.simulate()
            data = [tuple(j) for j in self.match.data_collection]

            filenumber = i

            filenumberstring = str(filenumber).zfill(4)
            filename = f"game_{filenumberstring}.pkl"
            while os.path.isfile(filename):
                filenumber += 1
                filenumberstring = str(filenumber).zfill(4)
                filename = f"game_{filenumberstring}.pkl"

            pickle.dump(data, open(GAMEDIR + "/" + filename, "wb"))

            del self.match
            self.match = SelfMatch()


if __name__ == "__main__":
    play = SelfPlay()
    play.start(50)
