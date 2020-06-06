import argparse

from Model.model import AZero
from azts.create_dataset import create_dataset
from azts import utility





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

    
    for _ in range(args.selflearnruns):

        create_dataset(yamlpaths=[args.player, args.player],
                       rollouts_per_move=args.rollouts_per_move,
                       num_of_parallel_processes=args.num_of_parallel_processes,
                       num_of_games_per_process=args.num_of_games_per_process,
                       fork_method=args.fork_method)

        model = utility.load_model(utility.load_player_conf(args.player))

        model.auto_run_training(max_iterations=args.iterations, \
                max_epochs=args.epochs)



