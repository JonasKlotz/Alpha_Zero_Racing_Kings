import stateMachine
import mockModel
import numpy as np

from config import *

# index of values that get stored
# in each edge
PPRIOR = 0
NCOUNT = 1
WACCUMVALUE = 2
QMEANVALUE = 3

EXPLORATION = 0.1
AMPLIFY_RESULT = 100 


class AztsNode():
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

            leaf = AztsNode(self.state_machine,
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

if __name__ == "__main__":
    state_machine = stateMachine.StateMachine()
    mock_model = mockModel.MockModel()
    node = AztsNode(state_machine, mock_model, state_machine.get_actual_position())
    for i in range(25):
        node.rollout()

    print(node)




