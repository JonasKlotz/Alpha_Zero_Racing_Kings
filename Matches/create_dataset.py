# pylint: disable=E0401
# pylint: disable=E0602
import random
import string
import multiprocessing
import os.path
import pickle
import argparse
from lib.logger import get_logger

from Player import config
from Azts import mock_model
from Azts import player
from Azts import utility
from Azts.config import GAMEDIR, DATASETDIR
from Matches import contest
from lib.timing import runtime_summary

log = get_logger("create_dataset")


# @timing
def create_dataset(yamlpaths,
                   rollouts_per_move,
                   num_of_parallel_processes,
                   num_of_games_per_process,
                   fork_method="spawn"):
    '''
    starts parallel training which creates
    many different game_[...].pkl files in
    GAMEDIR and then calls assemble_dataset
    which creates a dataset in DATASETDIR
    from all created games in GAMESDIR
    '''
    handle = utility.get_unused_match_handle(*tuple(yamlpaths))
    log.info(f"starting matches with handle {handle}")
    parallel_matches(yamlpaths=yamlpaths,
                     handle=handle,
                     rollouts_per_move=rollouts_per_move,
                     num_of_parallel_processes=num_of_parallel_processes,
                     num_of_games_per_process=num_of_games_per_process,
                     fork_method=fork_method)
    assemble_dataset(handle)
    return 0


def assemble_dataset(handle):
    '''
    collects all game files with a specific handle
    and assembles them into a dataset that is being
    stored in dataset directory
    :param str handle: handle string in the format
    of [NAME1].v[VERSION1].m[REVISION1]-\
            [NAME2].v[VERSION2].m[REVISION2]_\
            [8-DIGIT-RANDOM-KEY]
    only filenames that contain this string will
    be considered for assembling the dataset
    '''
    dataset = []
    counter = 0

    for filename in os.listdir(GAMEDIR):
        if f"game_{handle}" in filename and filename.endswith(".pkl"):
            filepath = os.path.join(GAMEDIR, filename)
            counter += 1
            game_data = pickle.load(open(filepath, "rb"))
            for i in game_data:
                dataset.append(i)
            log.info(f"added {filename} to dataset.")

    dataset_path = utility.get_unused_filepath(
        f"dataset_{handle}",
        DATASETDIR)

    pickle.dump(dataset, open(dataset_path, "wb"))
    log.info(f"saved data of {counter} games to {dataset_path}.")

    del dataset

    test_load = pickle.load(open(dataset_path, "rb"))
    log.info(f"verified integrity of file.\n"
             + f"dataset is of type {type(test_load)},\n"
             + f"has {len(test_load)} entries of type {type(test_load[0])}\n"
             + f"with {len(test_load[0])} entries of type\n"
             + f"{type(test_load[0][0])}, {type(test_load[0][1])}, "
             + f"{type(test_load[0][2])}.")

    return


def parallel_matches(yamlpaths,
                     handle,
                     rollouts_per_move,
                     num_of_parallel_processes,
                     num_of_games_per_process,
                     fork_method="spawn"):
    '''
    start parallel self play matches on different
    processes.
    :param list yamlpaths: list or tuple containing
    player configuration yaml paths
    :param str handle: identifier string for this
    self play session
    :param int rollouts_per_move: number of rollouts
    per move for each ai player. the higher this
    number, the longer a move and thus a game takes
    '''
    # according to
    # https://docs.python.org/3/library/multiprocessing.html
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method(fork_method)

    processes = []
    selfplays = []

    for i in range(num_of_parallel_processes):
        players = utility.load_players(*tuple(yamlpaths))
        selfplay = contest.Contest(
            player_one=players[0],
            player_two=players[1],
            rollouts_per_move=rollouts_per_move,
            game_id=handle,
            show_game=False)
        process = multiprocessing.Process(target=selfplay.start,
                                          args=(num_of_games_per_process,))
        process.start()
        processes.append(process)
        selfplays.append(selfplay)

    for i in processes:
        i.join()
    # returns after all parallel games are finished and
    # written to disk
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Multiprocessing generation of self-play "
                                     + "games. Each process generates games independently "
                                     + f"and each game is stored in {GAMEDIR}. Games are "
                                     + "collected after all processes are finished and "
                                     + "assembled into a single dataset, which is "
                                     + f"stored in {DATASETDIR}. The dataset is "
                                     + "being verified by loading it and providing "
                                     + "a print message with details before the "
                                     + "script terminates.")
    parser.add_argument("-p", "--num_of_parallel_processes",
                        type=int, default=2,
                        help="choose number of processes which generate "
                        + "games. The number of your CPU cores is a good "
                        + "starting point. Defaults to 2")
    parser.add_argument("-g", "--num_of_games_per_process",
                        type=int, default=10,
                        help="number of games to be created by each "
                        + "process. Defaults to 10")
    parser.add_argument("-r", "--rollouts_per_move",
                        type=int, default=100, help="number of "
                                                    + "rollouts that the engine performs while "
                                                    + "determinating a single move. Defaults to 100.")
    parser.add_argument("--fork_method", type=str,
                        default="spawn", help="depending on operating "
                                              + "system, different fork methods are valid for "
                                              + "multithreading. \"spawn\" has apparently the "
                                              + "widest compatibility. Other options are "
                                              + "\"fork\" and \"forkserver\". See "
                                              + "https://docs.python.org/3/library/multiprocessing.html "
                                              + "for details. Defaults to \"spawn\".")
    parser.add_argument("--player_one", type=str,
                        default="Player/default_config.yaml",
                        help="Player one configuration file. Is by default "
                             + "set to \"Player/default_config.yaml\".")
    parser.add_argument("--player_two", type=str,
                        default="Player/default_config.yaml",
                        help="Player two configuration file. Is by default "
                             + "set to \"Player/default_config.yaml\".")

    args = parser.parse_args()

    create_dataset(yamlpaths=[args.player_one, args.player_two],
                   rollouts_per_move=args.rollouts_per_move,
                   num_of_parallel_processes=args.num_of_parallel_processes,
                   num_of_games_per_process=args.num_of_games_per_process,
                   fork_method=args.fork_method)

    runtime_summary()
# pylint: enable=E0401
# pylint: enable=E0602
