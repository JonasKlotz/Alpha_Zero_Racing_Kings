from copy import copy
import io
# for svg rendering in pycharm
from PIL import Image
from cairosvg import svg2png
import random
import chess  # pip install python-chess
import chess.variant
import chess.engine

import chess.svg


class Game:
    std_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"
    valid_moves = []
    white_to_win = False
    black_to_win = False
    engine = None

    move_count = 0

    # init board either with fen or std fen string
    def __init__(self, fen=std_fen):
        self.board = chess.variant.RacingKingsBoard()
        self.player_to_move = 1
        print(self.board)

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

    def make_move(self, move):
        """
        must check if king landed on 8 rank
        Input:
            move: move taken by the current player  SAN Notation
        Returns:
            double: score of current player on the current turn
            int: player who plays in the next turn
        """
        self.board.push(move)
        self.move_count += 1
        self.player_to_move = (self.player_to_move + 1) % 2  # kann man bestimmt schöner machen
        self.is_won()

    def get_fen(self, ):
        """

        Returns:
            fen which will serve as an input to agent.predict
        """
        return self.board.board_fen()

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
        return self.board.is_variant_end()

    def is_draw(self):
        """
        Returns:
            boolean: True if game ended in a draw, False otherwise
        """
        return self.board.is_variant_draw()

    def get_score(self, player):  # 0 for black, 1 for white
        """
        Input:
            player: current player
        Returns:
            1-0, 0-1 or 1/2-1/2 if the game is over. Otherwise, the result is undetermined: *.
        """
        res = self.board.result()
        if res is '*':
            print("not Ended")
            return -1
        elif res == "1/2-1/2 ":
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
        rnd_move = random.choice(moves)
        self.make_move(rnd_move)

    def play_stockfish(self, limit=1.0):  # if not working make engine/stockfish-x86_64 executable
        """
        stockfish plays move
        :param limit:
        :return:
        """
        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci("engine/stockfish-x86_64")
        result = self.engine.play(self.board, chess.engine.Limit(time=limit))
        self.make_move(result.move)


game = Game()
print(game.board.turn)

while not game.is_ended():
    game.play_stockfish(limit=0.01)
game.show_game()
res = game.get_score(1)
print(res)
game.engine.quit()
