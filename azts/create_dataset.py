import multiprocessing
import os.path
import pickle

from azts import self_play
from azts.config import *


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
    multiprocessing.set_start_method("spawn") 
    num_of_processes = 6
    games_per_process = 3
    processes = []
    selfplays = []


    for i in range(num_of_processes):
        selfplay = self_play.SelfPlay(10)
        process = multiprocessing.Process(target = selfplay.start, \
                args = (games_per_process,))

        process.start()
        processes.append(process) 
        selfplays.append(selfplay)

    for i in processes:
        i.join()



    """
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

    """
