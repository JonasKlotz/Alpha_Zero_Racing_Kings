import multiprocessing
import os.path
import pickle

import argparse

from azts import self_play
from azts.config import *

parser = argparse.ArgumentParser(description = \
        "Multiprocessing generation of self-play " \
        + "games. Each process generates games independently " \
        + f"and each game is stored in {GAMEDIR}. Games are " \
        + "collected after all processes are finished and " \
        + "assembled into a single dataset, which is " \
        + f"stored in {DATASETDIR}. The dataset is " \
        + "being verified by loading it and providing " \
        + "a print message with details before the " \
        + "script terminates.")
parser.add_argument("-p", "--num_of_parallel_processes", \
        type = int, default = 2,
        help = "choose number of processes which generate " \
        + "games. The number of your CPU cores is a good " \
        + "starting point. Defaults to 2")
parser.add_argument("-g", "--num_of_games_per_process", \
        type = int, default = 10, \
        help = "number of games to be created by each " \
        + "process. Defaults to 10")
parser.add_argument("-r", "--rollouts_per_move", \
        type = int, default = 100, help = "number of " \
        + "rollouts that the engine performs while " \
        + "determinating a single move. Defaults to 100.")
parser.add_argument("--fork_method", type = str, \
        default = "spawn", help = "depending on operating " \
        + "system, different fork methods are valid for " \
        + "multithreading. \"spawn\" has apparently the " \
        + "widest compatibility. Other options are "\
        + "\"fork\" and \"forkserver\". See "\
        + "https://docs.python.org/3/library/multiprocessing.html "\
        + "for details. Defaults to \"spawn\".")
    


args = parser.parse_args()



filepath = os.path.join(GAMEDIR, "game_0000.pkl")
if os.path.exists(filepath):
    raise Exception("Games Directory not empty")

def unused_filename(num_of_games):
    filenumber = 0

    filenumberstring = str(filenumber).zfill(4)
    filename = f"dataset_{filenumberstring}_{num_of_games}_games.pkl"
    filepath = os.path.join(DATASETDIR, filename)
    while os.path.isfile(filepath):
        filenumber += 1
        filenumberstring = str(filenumber).zfill(4)
        filename = f"dataset_{filenumberstring}_{num_of_games}_games.pkl"
        filepath = os.path.join(DATASETDIR, filename)

    return filepath


if __name__ == "__main__":

    # according to
    # https://docs.python.org/3/library/multiprocessing.html
    multiprocessing.set_start_method(args.fork_method)
    processes = []
    selfplays = []


    for i in range(args.num_of_parallel_processes):
        selfplay = self_play.SelfPlay(args.rollouts_per_move)
        process = multiprocessing.Process(target = selfplay.start, \
                args = (args.num_of_games_per_process,))

        process.start()
        processes.append(process) 
        selfplays.append(selfplay)

    for i in processes:
        i.join()



    dataset = []


    counter = 0
    for filename in os.listdir(GAMEDIR):
        if "game_" in filename and filename.endswith(".pkl"):
            filepath = os.path.join(GAMEDIR, filename)
            counter += 1
            game_data = pickle.load(open(filepath, "rb"))
            for i in game_data:
                dataset.append(i)
            print(f"added {filename} to dataset.")

    dataset_path = unused_filename(counter)

    pickle.dump(dataset, open(dataset_path, "wb"))
    print(f"saved data of {counter} games to {dataset_path}.")

    del dataset

    test_load = pickle.load(open(dataset_path, "rb"))
    print(f"verified integrity of file.\n" \
            + f"dataset is of type {type(test_load)},\n" \
            + f"has {len(test_load)} entries of type {type(test_load[0])}\n" \
            + f"with {len(test_load[0])} entries of type\n" \
            + f"{type(test_load[0][0])}, {type(test_load[0][1])}, " \
            + f"{type(test_load[0][2])}.")

