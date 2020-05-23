import copy
import chess.variant
import numpy as np

from Interpreter import game
from Interface import TensorNotation as tn



class StateMachine():
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
        self.actual_game.board.set_fen(fen_state)
        self.reset_to_actual_game() 

    def move_index_to_fen(self, move_idx):
        return tn.tensor_indices_to_move(move_idx)

    def reset_to_actual_game(self):
        """
        after each rollout, set game
        to the actual state that the
        game is at this point of play
        """
        self.rollout_game = copy.deepcopy(self.actual_game)

    def get_legal_moves_from(self, position):
        fen = tn.tensor_to_fen(position)
        self.rollout_game.board = chess.variant.RacingKingsBoard(fen)

        return self.get_legal_moves()

    def get_legal_moves(self):
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

    def get_actual_position(self):
        return tn.fen_to_tensor(self.actual_game.board.fen())

    def get_player_color(self):
        if self.actual_game.board.turn:
            return 1
        return -1

    def rollout_idx_move(self, move_idx):
        move_fen = tn.tensor_indices_to_move(move_idx)
        try:
            self.rollout_game.make_move(move_fen)
        except:
            raise ValueError(f"rollout: move {move_fen} \
                    not possible in position \
                    {self.rollout_game.board.fen()}")
        return tn.fen_to_tensor(self.rollout_game.board.fen())

    def actual_idx_move(self, move_idx):
        move_fen = tn.tensor_indices_to_move(move_idx)
        self._actual_move(move_fen)

    def actual_fen_move(self, move_fen):
        self._actual_move(move_fen)

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

    def get_rollout_result(self):
        result = self.rollout_game.board.result()
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        return translate[result]

    def get_actual_result(self):
        result = self.actual_game.board.result()
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        return translate[result]

    def actual_has_ended(self):
        return self.actual_game.is_ended()

    def rollout_has_ended(self):
        return self.rollout_game.is_ended()


