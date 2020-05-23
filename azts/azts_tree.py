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
import state_machine as sm
import azts_node as an
import mock_model

from config import *



class AztsTree():
    """
    AztsTree represents the
    alpha zero search tree.
    """ 
    def __init__(self,
                 state_machine,
                 model,
                 color,
                 position,
                 runs_per_move=10):

        self.color = color

        self.state_machine = state_machine
        self.model = model
        self.runs_per_move = runs_per_move
        self._init_tree(position)

    def _init_tree(self, position=None):
        if position:
            self.state_machine.set_to_fen_position(position)
        self.root = an.AztsNode(self.state_machine,
                         self.model,
                         self.state_machine.get_actual_position(),
                         self.color)

    def __str__(self):
        return self.root.__str__()

    def make_move(self):
        move = ""
        if self.color == self.state_machine.get_player_color():
            self._tree_search(self.runs_per_move)
            move = self.root.get_move()
            self.state_machine.actual_fen_move(move)
        else:
            raise Exception("Other players turn")
        return move

    def get_policy_tensor(self):
        return self.root.get_policy_tensor()

    def receive_move(self, move):
        self.state_machine.actual_fen_move(move)

        del self.root
        self._init_tree()

    def get_position(self):
        return self.root.get_position()

    def _tree_search(self, runs=10):
        for i in range(runs):
            self.root.rollout()

    def _set_root_to(self, position):
        pass


def set_up():
    state_machine = sm.StateMachine()
    model = mock_model.MockModel()
    azts_tree = AztsTree(state_machine,
                model,
                WHITE,
                None,
                200)

    np.set_printoptions(suppress=True, precision=3)

    return state_machine, model, azts_tree


if __name__ == "__main__":
    state_machine, model, tree = set_up()

    print(f"Calculating first move...")
    time1 = time.time()
    first_move = tree.make_move()
    time2 = time.time()

    print(tree)
    mode = "selfplay" if SELFPLAY else "tournament"
    print(f"doing {tree.runs_per_move} rollouts " \
          + f"in {mode} mode took " \
          + f"{str(time2 - time1)[0:5]} seconds.\n")
    print(f"First move is {first_move}.")
