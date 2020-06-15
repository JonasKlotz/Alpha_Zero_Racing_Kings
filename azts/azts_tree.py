# pylint: disable=E0401
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
from Azts import state_machine
from Azts import azts_node
from Azts import mock_model 
from Azts.config import WHITE, ROLLOUTS_PER_MOVE, \
        EXPLORATION, ROLLOUT_PAYOFFS, HEAT


class AztsTree():
    """
    AztsTree represents the
    alpha zero search tree.
    :param str position: Game state in FEN-notation
    or None.
    """
    def __init__(self, \
                 model, \
                 color=WHITE, \
                 rollouts_per_move=ROLLOUTS_PER_MOVE, \
                 exploration=EXPLORATION, \
                 payoffs=ROLLOUT_PAYOFFS, \
                 heat=HEAT):

        self.color = color

        self.statemachine = state_machine.StateMachine()
        self.model = model
        self.rollouts_per_move = rollouts_per_move
        self.heat = heat

        # for initialising azts nodes:
        self.exploration = exploration
        self.payoffs = payoffs 

        self._init_tree()

    def _init_tree(self):
        '''
        keep this as separate function
        because it needs to be called
        after every move if tree is
        not reused
        '''
        self.root = azts_node.AztsNode(\
                statemachine=self.statemachine, \
                model=self.model, \
                color=self.color, \
                exploration=self.exploration, \
                payoffs=self.payoffs)

    def __str__(self):
        string = self.root.__str__()
        string += f"\n\nSettings:\n\tHeat:\t\t{self.heat}\n" \
                + f"\tExploration:\t{self.exploration}\n"
        return string

    def set_color(self, color):
        '''
        sets color
        '''
        self.color = color
        del self.root
        self._init_tree()

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
        '''
        calculate move
        :return str: move in uci notation
        '''
        move = ""

        if self.statemachine.actual_game_over():
            raise Exception("Game over")

        if self.color == self.statemachine.get_player_color():
            self._tree_search(self.rollouts_per_move)
            move, new_root = self.root.get_move(self.heat)
            if new_root is None:
                raise Exception("next node is none after make move")
            self.statemachine.actual_fen_move(move)
            self.root = new_root
        else:
            raise Exception("Other players turn")

        return move

    def get_move_statistics(self):
        '''
        return statistics about current state
        of azts tree search
        :return dict: dictionary containing
        entries about tree (max depth etc)
        and distribution (probability of
        best move etc).
        '''
        stats = self.root.get_move_statistics(self.heat)
        stats["settings"] = {}
        stats["settings"]["heat"] = self.heat
        stats["settings"]["exploration"] = self.exploration
        stats["settings"]["color"] = self.color
        stats["settings"]["payoffs"] = self.payoffs
        return stats


    def get_policy_tensor(self):
        '''
        :return np.array: move tensor with move
        distribution after alpha zero tree search
        '''
        return self.root.get_policy_tensor()

    def receive_move(self, move):
        '''
        update inner state with action of opponent
        :param str move: opponents move in uci notation
        '''
        if self.statemachine.actual_game_over():
            raise Exception("Game over")

        if self.color != self.statemachine.get_player_color():
            self.statemachine.actual_fen_move(move)
            new_root = self.root.select_node_with_move(move)
            self.root = new_root

            if new_root is None:
                # possible that this move was never simulated
                # this means no node for it exists yet
                del self.root
                self._init_tree() 

        else:
            raise Exception("My turn")

    def get_position(self):
        '''
        :return np.array: board position in tensor notation
        '''
        return self.root.get_position()

    def game_over(self):
        '''
        :return boolean: True if game is over
        '''
        return self.statemachine.actual_game_over()

    def game_result(self):
        '''
        :return int: 1 for white win, -1 for
        black win, 0 for running or draw
        '''
        return self.statemachine.get_actual_result()

    def game_state(self):
        '''
        :return int: enum types which are defined
        in config.py, determining the specific
        outcome (running, white wins, black wins,
        draw, draw by stale mate, draw by repetition,
        draw by two wins
        '''
        return self.statemachine.get_actual_state()

    def reset(self):
        '''
        re-initialise all stateful things
        '''
        del self.statemachine
        self.statemachine = state_machine.StateMachine()
        self._init_tree()

    def _tree_search(self, rollouts=10):
        '''
        :param int rollouts: number of rollouts to
        be performed on current game state
        '''
        for _ in range(rollouts):
            self.root.rollout()

    def _set_root_to(self, position):
        pass


def set_up(color=WHITE):
    '''
    helper function to initialise all
    data structures
    :return tuple: containing a state machine,
    a model and an azts tree that has been
    initialised with that state machine and model.
    '''
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = AztsTree(statemachine=statemachine, \
                model=model, \
                color=color, \
                rollouts_per_move=200)

    np.set_printoptions(suppress=True, precision=3)

    return statemachine, model, tree


if __name__ == "__main__":
    # pylint: disable=C0103
    statemachine, model, tree = set_up() 
    print(f"Calculating first move...")
    time1 = time.time()
    first_move = tree.make_move()
    time2 = time.time()

    print(tree)
    print(f"doing {tree.rollouts_per_move} rollouts " \
          + f"took {str(time2 - time1)[0:5]} seconds.\n")
    print(f"First move is {first_move}.")
    print("This might differ from the highest\n" \
            + "rated move because the actual move\n" \
            + "is randomly sampled from a distribution.")
    # pylint: enable=C0103
# pylint: enable=E0401
