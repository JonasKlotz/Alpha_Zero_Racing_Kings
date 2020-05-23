"""
azts module containing classes
for the alpha zero tree search.
the azts module is the entry
point and the actual tree is
built with node objects defined
in the node module.
TODO: Actually we dont need to
store the whole tensor of possible
moves in node.edges; legal moves
alone suffice. To make this work,
we'll have to reference
node.legal_move_indices to
reconstruct the corresponding move
"""
import time
import numpy as np
from azts import state_machine
from azts import azts_node
from azts import mock_model

from azts.config import *



class AztsTree():
    """
    AztsTree represents the
    alpha zero search tree.
    :param str position: Game state in FEN-notation
    or None.
    """ 
    def __init__(self,
                 statemachine,
                 model,
                 color,
                 runs_per_move=10):

        self.color = color

        self.statemachine = statemachine
        self.model = model
        self.runs_per_move = runs_per_move
        self._init_tree()

    def _init_tree(self):
        '''
        keep this as separate function
        because it needs to be called
        after every move if tree is
        not reused
        '''
        self.root = azts_node.AztsNode(self.statemachine, \
                         self.model, \
                         self.color) 

    def __str__(self):
        return self.root.__str__()

    def set_to_fen_state(self, fen_state):
        '''
        set internal game state to
        a state provided by fen_state
        :param str fen_state: fen notation
        of new state
        '''
        self.statemachine.set_to_fen_state(fen_state)
        del self.root
        self._init_tree()

    def make_move(self):
        move = ""

        if self.statemachine.actual_game_over():
            raise Exception("Game over") 

        if self.color == self.statemachine.get_player_color():
            self._tree_search(self.runs_per_move)
            move = self.root.get_move()
            self.statemachine.actual_fen_move(move)
        else:
            raise Exception("Other players turn")

        return move

    def get_policy_tensor(self):
        return self.root.get_policy_tensor()

    def receive_move(self, move):
        if self.statemachine.actual_game_over():
            raise Exception("Game over")

        if self.color != self.statemachine.get_player_color(): 
            self.statemachine.actual_fen_move(move)

            # TODO: check for reusability of current
            # tree. This should always be the case
            # if the opponents move leads to a follow-up
            # position
            del self.root
            self._init_tree()
        else:
            raise Exception("My turn")

    def get_position(self):
        return self.root.get_position()

    def game_over(self):
        return self.statemachine.actual_game_over()

    def game_result(self):
        return self.statemachine.get_actual_result()

    def game_state(self):
        return self.statemachine.get_actual_state()

    def _tree_search(self, runs=10):
        for i in range(runs):
            self.root.rollout()

    def _set_root_to(self, position):
        pass


def set_up(color = WHITE):
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = AztsTree(statemachine, \
                model, \
                color, \
                200)

    np.set_printoptions(suppress=True, precision=3)

    return statemachine, model, tree


if __name__ == "__main__":
    statemachine, model, tree = set_up()

    print(f"Calculating first move...")
    time1 = time.time()
    first_move = tree.make_move()
    time2 = time.time()

    print(tree)
    print(f"doing {tree.runs_per_move} rollouts " \
          + f"took {str(time2 - time1)[0:5]} seconds.\n")
    print(f"First move is {first_move}.")
