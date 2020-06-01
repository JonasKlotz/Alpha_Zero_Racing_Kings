# pylint: disable=E0401
# pylint: disable=E0602
import random
import string
import multiprocessing
import os.path
import pickle

import argparse


from Player import config
from azts import self_play
from azts import mock_model
from azts import player
from azts import utility
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
        type=int, default=2,
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
        + "widest compatibility. Other options are "\
        + "\"fork\" and \"forkserver\". See "\
        + "https://docs.python.org/3/library/multiprocessing.html "\
        + "for details. Defaults to \"spawn\".")
parser.add_argument("--player_one", type=str, \
        default="Player/default_config.yaml", \
        help="Player one configuration file") 
parser.add_argument("--player_two", type=str, \
        default="Player/default_config.yaml", \
        help="Player two configuration file")


args = parser.parse_args()

# from https://pynative.com/python-generate-random-string/
def random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def unused_filename(num_of_games, match_name):
    filenumber = 0

    filenumberstring = str(filenumber).zfill(4)
    filename = f"dataset_{match_name}_" \
            + f"{filenumberstring}_{num_of_games}_games.pkl"
    filepath = os.path.join(DATASETDIR, filename)
    while os.path.isfile(filepath):
        filenumber += 1
        filenumberstring = str(filenumber).zfill(4)
        filename = f"dataset_{match_name}_" \
                + f"{filenumberstring}_{num_of_games}_games.pkl"
        filepath = os.path.join(DATASETDIR, filename)

    return filepath


def create_dataset(yamlpath_one, yamlpath_two, fork_method="spawn"):
    handle = utility.get_unused_match_handle(yamlpath_one, yamlpath_two)
    

    return 0


def parallel_matches(yamlpaths, fork_method="spawn")
    # according to
    # https://docs.python.org/3/library/multiprocessing.html
    multiprocessing.set_start_method(args.fork_method)
    processes = []
    selfplays = []


    for i in range(args.num_of_parallel_processes):
        players = utility.load_players(*tuple(yamlpaths))


        selfplay = self_play.SelfPlay(\
                player_one=players[0], \
                player_two=players[1], \
                runs_per_move=args.rollouts_per_move, \
                game_id=game_id, \
                show_game=False)
        process = multiprocessing.Process(target = selfplay.start, \
                args = (args.num_of_games_per_process,))

        process.start()
        processes.append(process) 
        selfplays.append(selfplay)

    for i in processes:
        i.join()





if __name__ == "__main__":
    # pylint: disable=C0103
    conf_one = config.Config(args.player_one)
    conf_two = config.Config(args.player_two)

    player_names = [conf_one.name, conf_two.name]
    player_names.sort()


    match_name = f"{player_names[0]}-{player_names[1]}" 
    game_id = random_string(8)
    game_name = f"game_{game_id}_{match_name}"

    filepath = os.path.join(GAMEDIR, f"{game_name}_0000.pkl")
    while os.path.exists(filepath): 
        game_id = random_string(8)
        game_name = f"game_{game_id}_{match_name}"
        filepath = os.path.join(GAMEDIR, f"{game_name}_0000.pkl")

    # according to
    # https://docs.python.org/3/library/multiprocessing.html
    multiprocessing.set_start_method(args.fork_method)
    processes = []
    selfplays = []


    for i in range(args.num_of_parallel_processes):
        model = mock_model.MockModel()
        
        player_one = player.Player(model=model, \
                name=conf_one.name, \
                **(conf_one.player.as_dictionary()))

        player_two = player.Player(model=model, \
                name=conf_two.name, \
                **(conf_two.player.as_dictionary()))


        selfplay = self_play.SelfPlay(\
                player_one=player_one, \
                player_two=player_two, \
                runs_per_move=args.rollouts_per_move, \
                game_id=game_id, \
                show_game=False)
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
        if game_name in filename and filename.endswith(".pkl"):
            filepath = os.path.join(GAMEDIR, filename)
            counter += 1
            game_data = pickle.load(open(filepath, "rb"))
            for i in game_data:
                dataset.append(i)
            print(f"added {filename} to dataset.")

    dataset_path = unused_filename(counter, match_name)

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
    # pylint: enable=C0103

# pylint: enable=E0401
# pylint: enable=E0602
