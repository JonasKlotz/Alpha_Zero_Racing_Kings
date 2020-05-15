from Interpreter import game
from Interface import TensorNotation as tn

import chess.variant 
import numpy as np
import copy


class StateMachine():
    def __init__(self):
        self.actual_game = game.Game()
        self.rollout_game = game.Game()

        exemplary_move = self.actual_game.get_moves_observation()[0]
        self.move_shape = tn.move_to_tensor(exemplary_move).shape

        exemplary_pos = self.actual_game.board.fen()
        self.position_shape = tn.fen_to_tensor(exemplary_pos).shape

    def reset_to_actual_game(self):
        '''
        after each rollout, set game
        to the actual state that the
        game is at this point of play
        '''
        self.rollout_game = copy.deepcopy(self.actual_game)

    def get_legal_moves(self, position):
        '''
        Input is a position in tensor notation
        Output is a move tensor with all zeros
        except the legal moves which are ones.
        ''' 
        fen = tn.tensor_to_fen(position)
        self.rollout_game.board = chess.variant.RacingKingsBoard(fen) 

        moves = self.rollout_game.get_moves_observation()
        tensor = np.zeros(self.move_shape)
        
        for move in moves:
            tensor += tn.move_to_tensor(move)

        return tensor
    
    def get_legal_moves(self):
        moves = self.rollout_game.get_moves_observation()

        tensor = np.zeros(self.move_shape)
        
        for move in moves:
            tensor += tn.move_to_tensor(move)

        return tensor


    def get_new_position(self, old_pos, move_tensor):
        '''
        get new position from an
        old position and a move
        selected in old position
        '''
        old_fen = tn.tensor_to_fen(old_pos)
        self.rollout_game.board = chess.variant.RacingKingsBoard(old_fen)
        
        move_fen = tn.tensor_to_move(move_tensor) 

        try:
            self.rollout_game.make_move(move_fen)
        except:
            raise ValueError(f"move {move_fen} \
                    not possible in position {old_fen}")

        new_fen = rollout_game.board.fen()
        return tn.fen_to_tensor(new_fen)

    def get_actual_position(self):
        return tn.fen_to_tensor(self.actual_game.board.fen())

    def rollout_tensor_move(self, move_tensor):
        move_fen = tn.tensor_to_move(move_tensor)
        try:
            self.rollout_game.make_move(move_fen)
        except:
            raise ValueError(f"rollout: move {move_fen} \
                    not possible in position \
                    {self.rollout_game.board.fen()}")
        return tn.fen_to_tensor(self.rollout_game.board.fen())

    def actual_tensor_move(self, move_tensor):
        move_fen = tn.tensor_to_move(move_tensor)
        try:
            self.actual_game.make_move(move_fen)
        except:
            raise ValueError(f"actual move: move {move_fen} \
                    not possible in position \
                    {self.actual_game.board.fen()}")

    def actual_fen_move(self, move_fen):
        try:
            self.actual_game.make_move(move_fen)
        except:
            raise ValueError(f"actual move: move {move_fen} \
                    not possible in position \
                    {self.actual_game.board.fen()}")

    def get_rollout_result(self):
        result = self.rollout_game.board.result()
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        return translate[result] 

    def get_actual_result(self):
        result = self.actual_game.board.result()
        translate = {"*": 0, "1-0": 1, "0-1": -1, "1/2-1/2": 0}
        return translate[result] 


