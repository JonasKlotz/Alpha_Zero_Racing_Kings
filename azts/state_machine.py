# pylint: disable=E0401
'''
State machine to keep track of current
game state, to facilitate rollouts and
to translate from tensor to fen notation
and back.
'''
import copy
import chess.variant
import numpy as np

from Interpreter import game
from Interface import TensorNotation as tn

from azts.config import WHITE, BLACK, \
        RUNNING, BLACK_WINS, \
        WHITE_WINS, DRAW


class StateMachine():
    '''
    State machine to keep track of current
    game state, to facilitate rollouts and
    to translate from tensor to fen notation
    and back.
    State machine keeps track of TWO games:
    the actual game, from which rollouts can
    be made; and the state of a rollout.
    self.reset_to_actual_game() sets the
    rollout game back to the state of the
    actual game.

    All functions are polling the rollout
    game unless the function name contains
    "actual" in its name.
    '''
    def __init__(self):
        self.actual_game = game.Game()
        self.rollout_game = game.Game()

        exemplary_move = self.actual_game.get_moves_observation()[0]
        self.move_shape = tn.move_to_tensor(exemplary_move).shape

        exemplary_pos = self.actual_game.board.fen()
        self.position_shape = tn.fen_to_tensor(exemplary_pos).shape

    def set_to_fen_position(self, fen_position):
        self.actual_game.board.set_board_fen(fen_position)
        self.reset_to_actual_game()

    def set_to_fen_state(self, fen_state):
        '''
        setting new state (including players turn)
        for actual and rollout game
        :param int fen_state: state to set the game to
        '''
        self.actual_game.board.set_fen(fen_state)
        self.reset_to_actual_game() 

    def move_index_to_fen(self, move_idx):
        '''
        translate move index to uci notation
        :return str: move in uci notation
        '''
        return tn.tensor_indices_to_move(move_idx)

    def uci_to_move_idx(self, uci):
        '''
        translate uci move to indices of move
        in tensor notation
        '''
        return tn.move_to_tensor_indices(uci)

    def reset_to_actual_game(self):
        """
        after each rollout, set game
        to the actual state that the
        game is at this point of play
        """
        self.rollout_game = game.Game()
        current_state = self.actual_game.board.fen()
        self.rollout_game.board.set_fen(current_state)

    def get_legal_moves_from(self, position):
        '''
        get legal moves for a given
        position
        :param str position: position in
        fen notation
        '''
        fen = tn.tensor_to_fen(position)
        self.rollout_game.board = chess.variant.RacingKingsBoard(fen)

        return self.get_legal_moves()

    def get_legal_moves(self):
        '''
        get legal moves for current
        rollout state in tensor notation
        '''
        moves = self.rollout_game.get_moves_observation()

        legal_move_indices = np.zeros((len(moves), 3), dtype=np.uint16)

        for i, move in enumerate(moves):
            legal_move_indices[i] = tn.move_to_tensor_indices(move)

        return tuple(legal_move_indices.T)


    def get_new_position(self, old_pos, move_idx):
        """
        get new position from an
        old position and a move
        selected in old position
        """
        old_fen = tn.tensor_to_fen(old_pos)
        self.rollout_game.board = chess.variant.RacingKingsBoard(old_fen)

        move_fen = tn.tensor_indices_to_move(move_idx)

        try:
            self.rollout_game.make_move(move_fen)
        except:
            raise ValueError(f"move {move_fen} \
                    not possible in position {old_fen}")

        new_fen = self.rollout_game.board.fen()
        return tn.fen_to_tensor(new_fen)

    def get_position(self):
        '''
        :return np.array: position of rollout game
        in tensor notation
        '''
        return tn.fen_to_tensor(self.rollout_game.board.fen())

    def get_actual_position(self):
        '''
        :return np.array: position of actual game in tensor
        notation
        '''
        return tn.fen_to_tensor(self.actual_game.board.fen())

    def get_actual_fen_position(self):
        '''
        :return np.array: position of actual game in fen
        notation
        '''
        return self.actual_game.board.fen()

    def get_player_color(self):
        '''
        :return enum type: as defined in azts/config.py
        '''
        if self.actual_game.board.turn:
            return WHITE
        return BLACK

    def idx_move(self, move_idx):
        '''
        make a move in index notation in rollout game
        :param tuple: tuple with three np.arrays of
        shape (1,) denoting the index of a move in
        move tensor notation
        '''
        move_fen = tn.tensor_indices_to_move(move_idx)
        try:
            self.rollout_game.make_move(move_fen)
        except:
            raise ValueError(f"rollout: move {move_fen} \
                    not possible in position \
                    {self.rollout_game.board.fen()}")
        return tn.fen_to_tensor(self.rollout_game.board.fen())

    def actual_idx_move(self, move_idx):
        '''
        commits a move in tensor notation
        in both actual_game and rollout_game
        '''
        move_fen = tn.tensor_indices_to_move(move_idx)
        self._actual_move(move_fen)

    def actual_fen_move(self, move_fen):
        '''
        commits a move in uci notation
        in both actual_game and rollout_game
        '''
        try:
            self._actual_move(move_fen)
        except:
            raise ValueError(f"move {move_fen} \
                impossible in current position \
                {self.actual_game.board.fen()}")

    def _actual_move(self, move_fen):
        """
        commits a move in both actual_game and rollout_game
        :param str move_fen: move in uci notation
        """
        try:
            self.actual_game.make_move(move_fen)
        except:
            raise ValueError(f"actual move: move {move_fen} \
                not possible in position \
                {self.actual_game.board.fen()}")
        self.reset_to_actual_game()

    def get_result(self):
        """
        get result from rollout game
        """
        result = self.rollout_game.board.result()
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        return translate[result]

    def get_actual_result(self):
        '''
        get result from actual game
        :return int: -1 for black win, 1 for white
        win, 0 for running or draw
        '''
        result = self.actual_game.board.result()
        translate = {"*": RUNNING, "1-0": WHITE_WINS, \
                "0-1": BLACK_WINS, "1/2-1/2": DRAW}
        return translate[result]

    def actual_game_over(self):
        '''
        check if actual game is over
        :return boolean: True if actual game is over
        '''
        return self.actual_game.is_ended()

    def get_state(self):
        '''
        check state of rollout game according
        to enum type in config
        :return int: running, white win, black win,
        draw, draw by stale mate, draw by repetition,
        draw by two wins
        '''
        return self.rollout_game.get_game_state()

    def get_actual_state(self):
        '''
        check state of actual game according
        to enum type in config
        :return int: running, white win, black win,
        draw, draw by stale mate, draw by repetition,
        draw by two wins
        '''
        return self.actual_game.get_game_state()

    def game_over(self):
        """
        check if rollout game has ended
        """
        return self.rollout_game.is_ended()


# pylint: enable=E0401
