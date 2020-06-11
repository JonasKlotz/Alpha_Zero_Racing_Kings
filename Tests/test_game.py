import os

import chess

from azts.config import ROOTDIR
import platform

import pytest

# test policy in game.py
from game import Game

ENGINE = {"Linux": "stockfish-x86_64", \
          "Darwin": "stockfish-osx-x86_64", \
          "Windows": "stockfish-windows-x86_64.exe"}

PATH_TO_ENGINE = os.path.join(ROOTDIR, "Interpreter")
PATH_TO_ENGINE = os.path.join(PATH_TO_ENGINE, "Engine")
PATH_TO_ENGINE = os.path.join(PATH_TO_ENGINE, ENGINE[platform.system()])
ENGINE = {"Linux": "stockfish-x86_64", \
          "Darwin": "stockfish-osx-x86_64", \
          "Windows": "stockfish-windows-x86_64.exe"}
print(f"engine is at {PATH_TO_ENGINE}")


def test_policy():
    game = Game()
    pol = game.get_policy(PATH_TO_ENGINE)
    pol = game.normalize_policy(pol)
    assert (1.0 == sum(row[1] for row in pol))  # policy adds up to
    print(pol)


def test_policy_ingame():
    game = Game()
    for i in range(3):
        print(game.board)
        limit = 1
        pol = game.get_policy(PATH_TO_ENGINE, limit)
        # print("policy")
        # print(pol)

        pol = game.normalize_policy(pol)
        move = pol[0][0]

        print("best: ", pol[0][0])
        # print("worst: ", pol[-1][0])
        print(pol)
        print("engine")
        result = game.engine.play(game.board, chess.engine.Limit(time=limit))
        print(result.move.uci())

        game.make_move(move)
        # game.play_random_move()

    print(game.board)


def test_score_ingame():
    game = Game()
    p = 0
    while not game.is_ended():
        print(game.board)
        print("Score: ", game.get_score(path=PATH_TO_ENGINE))
        print("Eval: ", game.get_evaluation(path=PATH_TO_ENGINE))
        if p % 2 == 1:
            game.play_stockfish(path=PATH_TO_ENGINE)
        else:
            game.play_random_move()
        print()
        p += 1
    print(game.board)


if __name__ == "__main__":
    # test_policy()
    test_score_ingame()
