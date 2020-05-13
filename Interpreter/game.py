import sys
from copy import copy
import io
import random

# for svg rendering in pycharm
from PIL import Image
from cairosvg import svg2png
import chess  # pip install python-chess
import chess.variant
import chess.engine

import chess.svg


class Game:
    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.
    std_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"
    engine = None  # TODO does not belong in this class
    move_count = 0
    history = None  # Dict containing key : fen, value 0-3 for threefold repetition

    # init board either with fen or std fen string
    def __init__(self):
        self.board = chess.variant.RacingKingsBoard()
        self.player_to_move = 1  # 1 weiß,-1 schwarz
        self.result = None
        self.end = False
        self.draw = False
        self.history = {}
        self.history[self.board.fen()] = 1

        # print(self.board)

    def reset(self):
        """
        Resets to begin a new game
        :return Game: self
        """
        self.board = chess.variant.RacingKingsBoard()
        self.player_to_move = 1
        self.result = None
        self.end = False
        self.draw = False
        self.move_count = 0
        self.history.clear()
        self.history[self.std_fen] = 1
        return self

    def update(self, board):
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
        Returns:
             list <move> all legal moves
        """
        return list(self.board.legal_moves)

    def get_current_player(self):
        """
        Returns:
            int: current player idx
        """
        return self.board.turn  # white true, black false

    def get_movelist_size(self):
        """
        Returns:
            int: number of all possible actions
        """
        return len(list(self.board.legal_moves))

    def make_move(self, input):
        """
        must check if king landed on 8 rank
        Input:
            move: move taken by the current player  SAN Notation
        Returns:
            double: score of current player on the current turn
            int: player who plays in the next turn
        """

        try:
            move = self.board.parse_uci(input)  # UCI
        except:
            pass
        try:
            move = self.board.parse_san(input)  # SAN
        except:
            pass
        try:
            self.board.push(input)  # Move as chessmove
            self.after_mode()
            return
        except:
            pass
        try:
            self.board.push(move)
            self.after_mode()
            return
        except:
            self.show_game()
            print("move " + input + " illegal")
            self.end = True

    def after_mode(self, ):
        self.move_count += 1
        self.player_to_move *= -1

        try:
            if self.board.fen() in self.history:
                self.history[self.board.fen()] += 1
            else:
                self.history[self.board.fen()] = 1
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

    def is_won(self):
        """
           extracts last row looks if king on last_rank
        """
        return self.board.is_variant_win()

    def is_ended(self):
        """
        This method must return True if is_draw returns True
        Returns:
            boolean: False if game has not ended. True otherwise
        """
        self.end |= self.board.is_variant_end() or self.is_draw() or self.is_won()
        return self.end

    def is_draw(self):
        """
        Returns:
            boolean: True if game ended in a draw, False otherwise
        """
        self.draw |= self.board.is_variant_draw() or self.get_movelist_size() == 0
        if self.get_movelist_size() == 0: print("no valid move")
        if self.history[self.board.fen()] > 2: print("repetition")
        self.draw |= self.history[self.board.fen()] > 2
        return self.draw

    def get_score(self, player):  # 0 for black, 1 for white
        """
        Input:
            player: current player
        Returns:
            1, 0 or 1/2 if the game is over, depending on the player. Otherwise, the result is undetermined: *.
        """
        res = self.board.result()
        if res == '*':
            print("not Ended")
            return None
        if res == "1/2-1/2":
            return 0.5
        res = res.split('-')
        return float(res[player])

    def clone(self):
        """
        Returns:
            Game: a deep clone of current Game object
        """
        return copy(self)

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
            svg2png(bytestring=bytes(svg, 'UTF-8'), write_to=img)  # valid_moves = game.get_move_list()
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
            self.show_game()
            RuntimeError()
        self.make_move(rnd_move)


    def play_stockfish(self, limit=1.0):  # if not working make engine/stockfish-x86_64 executable
        """
        stockfish plays move
        :param limit:
        :return:
        """
        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci("Engine/stockfish-x86_64")
        result = self.engine.play(self.board, chess.engine.Limit(time=limit))
        self.make_move(result.move)


score = [0] * 3
for i in range(1):
    game = Game()
    while not game.is_ended():
        try:
            # game.play_stockfish(0.01)
            game.play_random_move()
            # game.make_move("h2h3")
        except:
            game.show_game()
            print("Fail")
            break
    s = game.get_score(1)
    print("s ", s)
    # score[int(s * 2)] += 1
game.show_game()
print(score)
# game.engine.close()

game = Game()
game.make_move("h2h3")
game.show_game()
