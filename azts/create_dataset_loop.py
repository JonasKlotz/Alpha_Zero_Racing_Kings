import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.timing import timing, runtime_summary, prettify_time
from lib.logger import get_logger
log = get_logger("Dataset")

from Model.model import AZero
from Player.config import Config

if __name__ == "__main__":
    SP_LENGTH = 1
    NUM_THREADS = 1

    @timing
    def create_dataset():
        os.system(
            "python3 -m azts.create_dataset --num_of_games_per_process=1 --num_of_parallel_processes=1")
        # create_dataset(yamlpaths=[args.player_one, args.player_two],
        #                rollouts_per_move=args.rollouts_per_move,
        #                num_of_parallel_processes=NUM_THREADS,
        #                num_of_games_per_process=SP_LENGTH,
        #                fork_method=args.fork_method,
        #                mockmodel=args.mock)

    conf = Config()
    model = AZero(conf)

    def create_dataset_loop():
        half_way_done = False
        # Generate
        i = 0
        while True:
            if half_way_done:  # expects new model to be ready soon
                log.info("Half-time done")
                start = time.perf_counter()
                if not model.new_model_available():
                    log.info("Waiting for newest model")
                while not model.new_model_available():
                    # NOTE: thread could just continue to create dataset (not const chunksize!)
                    time.sleep(1)
                log.info("Found new model")
                elapsed = time.perf_counter() - start
                log.info("Thread blocked for %s.", prettify_time(elapsed))
                model.restore_latest_model()
            log.info("Beginning Self-play iteration {}, "
                     "game chunk-size: {}".format(i, SP_LENGTH))
            i += 1
            create_dataset()
            half_way_done = ~half_way_done
            runtime_summary()
    create_dataset_loop()
