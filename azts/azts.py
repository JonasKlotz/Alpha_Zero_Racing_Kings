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
import stateMachine as sm

# dimensions of mock object tensors
DIMENSIONS = (8, 8, 64)

# index of values that get stored
# in each edge
PPRIOR = 0
NCOUNT = 1
WACCUMVALUE = 2
QMEANVALUE = 3

MOVE_DTYPE = np.uint8
POS_DTYPE = np.uint8
# np.float16:
# no overflow in -65500 .. 65500
EDGE_DTYPE = np.float16
IDX_DTYPE = np.uint16

SELFPLAY = False
EXPLORATION = 0.1
AMPLIFY_RESULT = 100

WHITE = 1
BLACK = -1


class MockTranslator():
    def get_legal_moves(self, position):
        legal_moves = np.random.rand(*DIMENSIONS)
        legal_moves[legal_moves > 0.5] = 1
        legal_moves[legal_moves < 0.6] = 0
        return legal_moves


class MockModel():
    def inference(self, position):
        policy = np.random.rand(*DIMENSIONS)
        evaluation = np.random.rand(1)[0] - 0.5

        return (policy, evaluation)


class Azts():
    """
    Azts represents the
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
        self.root = Node(self.state_machine,
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


class Node():
    """
    Node of the Alpha Zero Tree Search.
    Each node represents a game state
    and holds all possible moves as
    edges to other states. Edges keep
    track of the variables of the
    algorithm: N, W, Q, P.
    """

    # pylint: disable=C0326
    def __init__(self, state_machine, model, position, color=WHITE):

        self.color = color

        self.state_machine = state_machine
        self.model = model
        self.evaluation = 0
        self.endposition = False

        self.position_shape = position.shape
        self.position = self._compress_indices(position)

        self.move_shape = state_machine.move_shape
        # in selfplay, state machine tracks state
        # in tournament, we expand the tree independently
        # and only set state at leafs to a position
        self.legal_move_indices = \
            state_machine.get_legal_moves() if SELFPLAY else \
                state_machine.get_legal_moves_from(position)

        num_of_legal_moves = len(self.legal_move_indices[0])
        self.children = [None] * num_of_legal_moves

        if num_of_legal_moves == 0:
            # reached end of game
            result = state_machine.get_rollout_result()
            self.evaluation = result * self.color * AMPLIFY_RESULT
            self.endposition = True

        else:
            # expansion of node
            policy, self.evaluation = model.inference(position)

            # initialise tensor to hold
            # 4 values per edge
            entries_per_edge = 4
            self.edges = np.zeros(
                (num_of_legal_moves, entries_per_edge),
                dtype=EDGE_DTYPE)

            # only store prior values of legal moves
            self.edges[:, PPRIOR] = policy[self.legal_move_indices]

    def __str__(self):
        metrics, tree_string = self._print_tree(0)
        num_of_nodes = metrics[0] + metrics[1] + metrics[2]
        avg_num_of_move_possibilities = metrics[5] / num_of_nodes
        metric_string = "\nMORE than " if metrics[3] else "\n"
        metric_string += f"{num_of_nodes} nodes in total:\n" \
                         + f"\t{metrics[1]} leaf nodes\n" \
                         + f"\t{metrics[2]} end positions\n" \
                         + f"\t{metrics[0]} normal tree nodes\n" \
                         + f"\t{metrics[4]} was maximal tree depth\n" \
                         + f"\t{str(avg_num_of_move_possibilities)[0:5]}" \
                         + f" was average number of move possibilities per move"

        return tree_string + metric_string

    def _print_tree(self, level):
        l_shape = chr(9492)
        t_shape = chr(9500)
        i_shape = chr(9474)
        minus_shape = chr(9472)

        if level > 1000:
            return (0, 0, 0, 1, level, 0), "..."
        elif self.endposition:
            return (0, 0, 1, 0, level, 0), "end, {:0.3f}".format(self.evaluation)
        elif not any(self.children):
            return (0, 1, 0, 0, level, len(self.children)), \
                   "leaf, {:0.3f}".format(self.evaluation)

        rep = []
        metrics = (1, 0, 0, 0, level, len(self.children))
        maxlevel = level
        for i, j in enumerate(self.children):
            if j:
                child_metrics, string = j._print_tree(level + 1)
                metrics = [k + l for k, l in zip(metrics, child_metrics)]
                maxlevel = max(maxlevel, child_metrics[4])
                rep.append(str(i) + ": " + string)

        metrics[4] = maxlevel

        for i, j in enumerate(rep):
            # not last element
            if i < len(rep) - 1:
                if "\n" in j:
                    j = j.replace("\n", "\n" + i_shape + " ")
                j = t_shape + minus_shape + j

            else:
                if "\n" in j:
                    j = j.replace("\n", "\n" + "  ")
                j = l_shape + minus_shape + j
            rep[i] = j

        repstr = ""
        for i in rep:
            repstr += i + "\n"

        repstr = "node, {:0.3f}\n".format(self.evaluation) + repstr
        return metrics, repstr

    def _compress_indices(self, tensor):
        """
        tensor is a positional tensor with
        entries 0 and 1. compress indices
        extracts the indices of entries 1 in
        tensor and converts them into reduced
        data type
        """
        indices = np.where(tensor == 1)
        compressed = []
        for i in indices:
            compressed.append(i.astype(IDX_DTYPE))

        return tuple(compressed)

    def get_policy_tensor(self):
        num_of_rollouts = self.edges[:, NCOUNT].sum()
        num_of_rollouts = max(1, num_of_rollouts)
        policy_weights = self.edges[:, NCOUNT] / num_of_rollouts
        policy_tensor = np.zeros(self.move_shape, EDGE_DTYPE)
        policy_tensor[self.legal_move_indices] = policy_weights
        return policy_tensor

    def get_move(self):
        i = np.argmax(self.edges[:, NCOUNT])
        i = self._legal_to_total_index(i)
        return self.state_machine.move_index_to_fen(i)

    def rollout(self, level=0):
        """
        recursive traversal of game tree
        updates P, N, W, Q of all travelled
        edges
        """
        if self.endposition:
            # terminate recursion
            # if end state is reached
            return self.evaluation

        i = self._index_of_best_move()
        next_node = self.children[i]

        next_position = None

        move_idx = self._legal_to_total_index(i)
        if SELFPLAY:
            # in self-play, we let the state
            # machine keep track of every move
            # to catch move-repetitions as draws
            # so that we get a realistic data set
            next_position = self.state_machine.rollout_idx_move(move_idx)

        evaluation = 0

        if next_node is None:
            # terminate recursion
            # on leaf expansion
            if not SELFPLAY:
                # if not in self-play, we do not use
                # the state machine, because all
                # the conversions tensor-fen are
                # quite expensive and evaluation
                # depends on the model anyways
                current_position = self._position_as_tensor()
                next_position = self.state_machine.get_new_position(
                    current_position, move_idx)

            leaf = Node(self.state_machine,
                        self.model,
                        next_position,
                        self.color)
            evaluation = leaf.evaluation
            self.children[i] = leaf

        else:
            # recursion
            evaluation = next_node.rollout(level + 1)

        # calculate edge stats
        count = self.edges[i][NCOUNT] + 1
        accum = self.edges[i][WACCUMVALUE] + evaluation

        # update edge stats
        self.edges[i][QMEANVALUE] = accum / count
        self.edges[i][NCOUNT] = count
        self.edges[i][WACCUMVALUE] = accum

        if SELFPLAY and level == 0:
            self.state_machine.reset_to_actual_game()

        return evaluation

    def _index_of_best_move(self):
        """
        returns the index of the best move
        which then can be used to access the
        4 values P, N, W, Q
        """

        U = self.edges[:, PPRIOR] / (self.edges[:, NCOUNT] + 1)
        U *= EXPLORATION
        Q = self.edges[:, QMEANVALUE]
        best_move_index = (Q + U).argmax()

        return best_move_index

    def _legal_to_total_index(self, index):
        """
        translates index to a move
        in the legal move selection
        to an index to the same move
        in all moves
        """
        return tuple(np.array(self.legal_move_indices).T[index])

    def _position_as_tensor(self):
        """
        this node represents a game
        position, which is stored in
        compressed form.
        _position_as_tensor returns
        the decompressed tensor notation
        of that position.
        """
        tensor = np.zeros(self.position_shape, dtype=POS_DTYPE)
        tensor[self.position] = 1
        return tensor

    def _move_as_tensor(self, move_index):
        """
        takes a move index which refers
        to an edge in edges and converts
        this single move to tensor notation.
        tensor is of data type POS_DTYPE,
        because entries are only zeros and
        one 1 for the selected move.
        """
        tensor = np.zeros(self.move_shape, dtype=POS_DTYPE)
        i = self._legal_to_total_index(move_index)
        tensor[i] = 1
        return tensor

    def _edges_as_tensor(self):
        """
        return move recommendation
        distribution over all legal
        moves in one tensor according
        to tensor notation of moves.
        """
        tensor = np.zeros(self.move_shape, dtype=MOVE_DTYPE)
        # TODO: actually put values there.
        return tensor

    def get_position(self):
        board = np.zeros(self.position_shape, dtype=POS_DTYPE)
        board[self.position] = 1
        return board

    # pylint: enable=C0326


def set_up():
    state_machine = sm.StateMachine()
    model = MockModel()
    azts = Azts(state_machine,
                model,
                WHITE,
                None,
                200)

    np.set_printoptions(suppress=True, precision=3)

    return state_machine, model, azts


if __name__ == "__main__":
    state_machine, model, tree = set_up()

    print(f"Calculating first move...")
    time1 = time.time()
    print(f"First move is {tree.make_move()}.")
    time2 = time.time()

    print(tree)
    mode = "selfplay" if SELFPLAY else "tournament"
    print(f"doing {tree.runs_per_move} rollouts " \
          + f"in {mode} mode took " \
          + f"{str(time2 - time1)[0:5]} seconds.\n")
