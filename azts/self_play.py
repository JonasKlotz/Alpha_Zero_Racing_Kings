# pylint: disable=E0401
# pylint: disable=E0602
"""
This module simulates many games of RacingsKings.
"""

import os.path
import time
import pickle
from azts import self_match
from azts.config import *

import lib.timing
from lib.logger import get_logger
log = get_logger(__name__)


def unused_filename(i=0):
    '''
    function to find the lowest unused
    filename within games folder according
    to naming scheme "game_0000.pkl"
    '''
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
    '''
    selfplay is initialized with the number of
    rollouts that the matching ai player are
    using per move.
    the number of game simulations is determined
    by the parameter in function start() which
    actually starts the series of matches.
    After each match, the match data is written
    to a separate file which facilitates
    parallelisation of creating data for many
    matches.
    '''

    def __init__(self, model, runs_per_move=RUNS_PER_MOVE):
        self.model = model
        self.match = self_match.SelfMatch(
            self.model, runs_per_move=runs_per_move)
        self.runs_per_move = runs_per_move

    def start(self, iterations=10):
        '''
        start a series of matches. match data
        for each match is written to a separate
        file in the games folder as defined in
        config.
        :param int iterations: number of matches
        to be simulated
        '''
        for i in range(iterations):
            self.match.simulate()
            data = [tuple(j) for j in self.match.data_collection]

            filepath = unused_filename(i)

            pickle.dump(data, open(filepath, "wb"))

            del self.match
            self.match = self_match.SelfMatch(
                self.model, runs_per_move=self.runs_per_move)


if __name__ == "__main__":

    import os

    MAX_RUNS = 4
    SP_LENGTH = 4

    from Model.model import AZero

    model = AZero()
    play = SelfPlay(model)

    passed_half = False
    # Generate
    for i in range(MAX_RUNS):
        if passed_half:  # expects new model to be ready soon
            start = time.perf_counter()
            log.info("Waiting for newest model")
            while not model.new_model_available():
                # NOTE: thread could just continue to create dataset (not const chunksize!)
                time.sleep(1)
            log.info("Found new model")
            elapsed = time.perf_counter - start
            log.info("Thread blocked for {}".format(timing.prettify(elapsed)))
            model.restore_latest_model()
        log.info("Beginning Self-play iteration {}/{}, \
            game chunk-size: {}".format(i, MAX_RUNS, SP_LENGTH))
        play.start(SP_LENGTH)
        passed_half = not passed_half


# pylint: enable=E0401
# pylint: enable=E0602
