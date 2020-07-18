import argparse
from lib.logger import get_logger

from Matches.create_dataset import  assemble_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--handle", type=str,
                        default="StockingFish-StockingFish")

    parser.add_argument("-g", "--max_games",
                        type=int, default=30000)

    args = parser.parse_args()

    assemble_dataset(args.handle, max_games=args.max_games)
