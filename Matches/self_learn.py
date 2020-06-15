import argparse
import copy
import multiprocessing
from lib.logger import get_logger

from Model.model import AZero
from Azts import utility
from Matches import contest
from Matches.create_dataset import assemble_dataset

log = get_logger("self_learn")

def parallel_matches_with_preloaded_model(yamlpath, \
        model, \
        handle, \
        rollouts_per_move, \
        num_of_parallel_processes, \
        num_of_games_per_process, \
        fork_method="spawn"):
    '''
    this version of parallel matches takes a preloaded
    model as input and creates a deepcopy for each thread
    to achieve real parallel processing without loading
    the model n-times from the mlflow server
    '''

    # according to
    # https://docs.python.org/3/library/multiprocessing.html
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method(fork_method)


    processes = []
    selfplays = []

    for _ in range(num_of_parallel_processes):
        # create a model copy for each parallel
        # process
        modelcopy = copy.deepcopy(model)
        players = [utility.load_player_with_model(model=modelcopy, \
                config=utility.load_player_conf(yamlpath)) for _ \
                in range(2)]

        # self play, so both sides are played by
        # same player
        selfplay = contest.Contest(
            player_one=players[0],
            player_two=players[1],
            rollouts_per_move=rollouts_per_move,
            game_id=handle,
            show_game=False)
        selfplays.append(selfplay)

    for i in selfplays:
        process = multiprocessing.Process(target=i.start, \
                args=(num_of_games_per_process,))
        process.start()
        processes.append(process)

    for i in processes:
        i.join()

    # just to make sure: clean up
    for i in selfplays:
        del i






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a dataset and perform " \
            + "gradient descent on it in a loop for -s number of steps.")
    parser.add_argument("-p", "--num_of_parallel_processes", \
            type=int, default=2, \
            help="choose number of processes which generate " \
            + "games. The number of your CPU cores is a good " \
            + "starting point. Defaults to 2")
    parser.add_argument("-g", "--num_of_games_per_process", \
            type=int, default=10, \
            help="number of games to be created by each " \
            + "process. Defaults to 10")
    parser.add_argument("-r", "--rollouts_per_move", \
            type=int, default=100, help="number of " \
            + "rollouts that the engine performs while " \
            + "determinating a single move. Defaults to 100.")
    parser.add_argument("--fork_method", type=str, \
            default="spawn", help="depending on operating " \
            + "system, different fork methods are valid for " \
            + "multithreading. \"spawn\" has apparently the " \
            + "widest compatibility. Other options are " \
            + "\"fork\" and \"forkserver\". See " \
            + "https://docs.python.org/3/library/multiprocessing.html " \
            + "for details. Defaults to \"spawn\".")
    parser.add_argument("--player", type=str, \
            default="Player/default_config.yaml", \
            help="Player configuration file. Is by default " \
            + "set to \"Player/default_config.yaml\".") 
    parser.add_argument("-i", "--iterations", type=int, \
            default=5, \
            help="number of iterations to train a model on " \
            + "one given dataset. Default: 5.")
    parser.add_argument("-e", "--epochs", type=int, \
            default=50, \
            help="number of epochs per training iteration. " \
            + "Default: 50.")
    parser.add_argument("-s", "--selflearnruns", type=int, \
            default=5, \
            help="number of self learn runs to perform. " \
            + "Default: 5.")

    args = parser.parse_args()

    # first load model from mlflow server
    model = utility.load_model(utility.load_player_conf(args.player))

    for _ in range(args.selflearnruns):
        handle = utility.get_unused_match_handle(args.player, args.player)
        log.info(f"starting matches with handle {handle}")

        parallel_matches_with_preloaded_model(yamlpath=args.player, \
                model=model, \
                handle=handle, \
                rollouts_per_move=args.rollouts_per_move, \
                num_of_parallel_processes=args.num_of_parallel_processes, \
                num_of_games_per_process=args.num_of_games_per_process, \
                fork_method=args.fork_method)

        assemble_dataset(handle=handle)

        model.auto_run_training(max_iterations=args.iterations, \
                max_epochs=args.epochs)



