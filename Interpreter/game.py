"""
provides functions to use stockfish with
python-chess library
"""

import sys
from copy import copy
import io
import random
import numpy as np

import chess  # pip install python-chess
import chess.variant
import chess.engine
import chess.svg

# for svg rendering in pycharm
from cairosvg import svg2png
from PIL import Image

from Interface.TensorNotation import DATATYPE, move_to_tensor_indices
from azts.config import *

class Game:
    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.
    std_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"
    engine = None
    history = None  # Dict containing key : fen, value 0-3 for threefold repetition

    # init board either with fen or std fen string
    def __init__(self):
        self.board = chess.variant.RacingKingsBoard()
        self.end = False
        self.draw = False
        self.history = {}
        self.history[self.board_fen_hash()] = 1
        self.state = RUNNING

        # print(self.board)

    def reset(self):
        """
        Resets to begin a new game
        :return Game: self
        """
        self.board = chess.variant.RacingKingsBoard()
        self.end = False
        self.draw = False
        self.history.clear()
        self.history[self.std_fen] = 1
        return self

    def update(self, board):  # TODO: Überarbeiten
        """
        Like reset, but resets the position to whatever was supplied for board
        :param chess.Board board: position to reset to
        :return game: self
        """
        self.board = chess.variant.RacingKingsBoard(board)
        self.move_count = 0
        return self

    def get_move_list(self):
        """
        :returns:
             list <move> all legal moves
        """
        return list(self.board.legal_moves)

    def get_current_player(self):
        """
        :Returns:
            int: current player idx
        """
        return self.board.turn  # white true, black false

    def get_movelist_size(self):
        """
        :Returns:
            int: number of all possible actions
        """
        return len(list(self.board.legal_moves))

    def make_move(self, input):
        """
        must check if king landed on 8 rank
        :input:
            move: move taken by the current player  SAN Notation
        :Returns:
            double: score of current player on the current turn
            int: player who plays in the next turn
        """

        try:
            move = self.board.parse_uci(input)  # UCI
            self.board.push(move)
            self.after_move()
            return
        except:
            pass
        try:
            self.board.push(input)  # Move as chessmove
            self.after_move()
            return
        except:
            pass

        try:
            move = self.board.parse_san(input)  # SAN
            self.board.push(move)
            self.after_move()
            return
        except:
            raise ValueError(f"move {input} illegal")

    def board_fen_hash(self):
        return self.board.fen().split(' ')[0]

    def after_move(self, ):

        board_fen = self.board_fen_hash()
        try:
            if board_fen in self.history:
                self.history[board_fen] += 1
            else:
                self.history[board_fen] = 1
        except:
            print("Unexpected error:", sys.exc_info()[0])

        self.end |= self.board.is_variant_end() or self.is_draw() or self.is_won()

    def get_observation(self):
        """

        Returns:
            fen which will be translated to tensor
        """
        return self.board.board_fen()

    def get_moves_observation(self):
        """

        :return: List of moves as UCI String
        """
        ml = self.get_move_list()
        return [move.uci() for move in ml]

    def get_game_state(self):
        self.is_draw()
        self.is_ended()
        return self.state

    def is_won(self):
        """
           extracts last row looks if king on last_rank
        """
        won = self.board.is_variant_win()
        if won:
            if self.board.result() == "1-0":
                self.state = WHITE_WINS
            if self.board.result() == "0-1":
                self.state = BLACK_WINS
        return won

    def is_ended(self):
        """
        This method must return True if is_draw returns True
        Returns:
            boolean: False if game has not ended. True otherwise
        """
        ended = self.board.is_variant_end()
        if ended:
            if self.board.result() == "1-0":
                self.state = WHITE_WINS
            if self.board.result() == "0-1":
                self.state = BLACK_WINS
        self.end |= ended or self.is_draw() or self.is_won()
        return self.end

    def is_draw(self):
        """
        Returns:
            boolean: True if game ended in a draw, False otherwise
        """
        self.draw |= self.board.is_variant_draw() or self.get_movelist_size() == 0

        white_finish = self.board.king(True) > 55
        black_finish = self.board.king(False) > 55
        both_finish = white_finish and black_finish

        if self.get_movelist_size() == 0:
            if both_finish:
                self.state = DRAW_BY_TWO_WINS
            else:
                self.state = DRAW_BY_STALE_MATE
            self.end = True

        try:
            if self.history[self.board_fen_hash()] > 2:
                self.state = DRAW_BY_REP
                self.end = True
                self.draw = True
        except:
            # to keep performance, we dont check if self.board.fen
            # is in self.history.keys(). we just try and pass
            # if it iss
            pass

        return self.draw

    def clone(self):
        """
        Returns:
            Game: a deep clone of current Game object
        """
        return copy(self)

    def render_game(self):
        svg = chess.svg.board(board=self.board)
        img = io.BytesIO()
        svg2png(bytestring=bytes(svg, 'UTF-8'), write_to=img)
        img = Image.open(img)
        return img

    def show_game(self, save=False, path=None):
        """
        converts game svg to png
        if safe = true safes it in output path
        """
        svg = chess.svg.board(board=self.board)

        if save:
            svg2png(bytestring=bytes(svg, 'UTF-8'), write_to=path)
        else:
            img = io.BytesIO()
            # valid_moves = game.get_move_list()
            svg2png(bytestring=bytes(svg, 'UTF-8'), write_to=img)
            img = Image.open(img)
            img.show()
            img.close()

    def play_random_move(self):
        """
        plays random move
        """
        moves = self.get_move_list()
        try:
            rnd_move = random.choice(moves)
        except:
            # self.show_game()
            RuntimeError()
        self.make_move(rnd_move)

    # if not working make engine/stockfish-x86_64 executable
    def play_stockfish(self, limit=1.0, path="Engine/stockfish-x86_64"):
        """
        stockfish plays move
        :param path: gives path to Stockfish Engine. Defaults to this (sub)directory.
        :param limit:
        :return:
        """
        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
        result = self.engine.play(self.board, chess.engine.Limit(time=limit))
        self.make_move(result.move)

    def get_policy(self, path="Engine/stockfish-x86_64", time_limit=0.1, depth_limit=None):
        """

        :rtype: [[UCI String][Centipawnscore]]
        :return: Policy as list of lists
        """
        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)

        policy = []

        for move in self.get_move_list():
            self.board.push(move)

            try:
                info = self.engine.analyse(self.board, chess.engine.Limit(
                    time=time_limit, depth=depth_limit))
                t = [move.uci(), - info["score"].relative.score(mate_score=100000)]
                policy.append(t)
            except:
                print("move is null", move.uci())
                t = [move.uci(), 0]

            self.board.pop()

        return policy

    def get_score(self, path="Engine/stockfish-x86_64"):
        """
        :return: returns winning probabilty in intervall between -1,1
        """
        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)

        try:
            info = self.engine.analyse(self.board, chess.engine.Limit(time=0.01))
            centipawn = info["score"].white().score(mate_score=100000) / 100
            # calculate winning probability
            # https://www.chessprogramming.org/Pawn_Advantage,_Win_Percentage,_and_Elo
            winning_probability = 1 / (1 + pow(10, -centipawn / 4))
            winning_probability = (winning_probability - 0.5) * 2.  # scale to [-1, 1]
            return winning_probability
        except:
            print(self.board)

            raise RuntimeError("coudlt calculate probability")  # TODO: LOGG

    def get_evaluation(self, path="Engine/stockfish-x86_64", time_limit=0.1, depth_limit=None):
        """
        :return: int with position evaluation score
        """

        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)

        try:
            info = self.engine.analyse(self.board, chess.engine.Limit(
                time=time_limit, depth=depth_limit))
            centipawn = info["score"].relative.score(mate_score=100000)  # / 100
            return centipawn
        except:
            print(self.board)
            # self.show_game()
            raise RuntimeError("coudlt calculate evaluation score")  # TODO: LOGG

    def normalize_policy(self, policy, x=-1):
        """
        :param policy: policy as from game.get_policy
        :param x: value how many values of the policy you want to keep
        :return: sorted normalized cut policy between [0,1]
        """
        policy.sort(key=lambda x: x[1], reverse=True)  # sort policy
        if x > 0:
            policy = policy[:x]
        s = abs(sum(row[1] for row in policy))
        for i in range(len(policy)):
            policy[i][1] /= s

        return policy

    def normalize_policy_zero_one(self, policy, x=-1):
        """
        :param policy: policy as from game.get_policy
        :param x: value how many values of the policy you want to keep
        :return: sorted normalized cut policy between [0,1]
        """
        policy.sort(key=lambda x: x[1], reverse=True)  # sort policy
        n = len(policy)
        maximum = policy[0][1]
        minimum = policy[-1][1]

        if x > 0:
            policy = policy[:x]
        for i in range(n):
            policy[i][1] += abs(minimum)
        s = abs(sum(row[1] for row in policy))
        for i in range(n):
            policy[i][1] /= s

        return policy

    # def evaluate_position(position, path="Engine/stockfish-x86_64"):
    #     """
    #     :param np.array: current game position
    #         in tensor notation
    #     :return: int with position evaluation score
    #     """
    #     g = Game()
    #     g.board = tensor_to_fen(position)
    #     return g.get_evaluation(path)

    # def get_policy_from_position(position):
    #     """
    #     Returns normalized policy for a given position in tensor notation
    #     :param position: np.array: current game position
    #         in tensor notation
    #     :return: normalized policy tensor
    #     """
    #     g = Game()
    #     g.board = tensor_to_fen(position)
    #     return policy_to_tensor(normalize_policy(g.get_policy()))

    def policy_to_tensor(policy):
        """
        :param policy: policy as from game.get_policy
        :return: tensor regarding the policy
        """
        tensor = np.zeros((8, 8, 64)).astype(DATATYPE)
        for uci, prob in policy:
            index = move_to_tensor_indices(uci)
            tensor[index[0], index[1], index[2]] = prob

        return tensor

if __name__ == "__main__":
    # game.make_move(policy[3][0])
    # print(game.get_scoring())
    game = Game()
    i = 0
    while not game.is_ended():
        game.play_random_move()
        game.get_score()
        print(game.get_score(), i)
        i += 1

    game.show_game()
"""
asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())

except:
print()
print("Unexpected error:", sys.exc_info()[0])
"""
