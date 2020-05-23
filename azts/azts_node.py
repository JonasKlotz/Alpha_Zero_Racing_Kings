"""
Node of the Alpha Zero Tree Search.
Each node represents a game state
and holds all possible moves as
edges to other states. Edges keep
track of the variables of the
algorithm: N, W, Q, P.
"""
import numpy as np

from azts import state_machine
from azts import mock_model

from azts.config import WHITE, POS_DTYPE,\
        EDGE_DTYPE, IDX_DTYPE, SELFPLAY, \
        AMPLIFY_RESULT, EXPLORATION

# pylint: disable=W0621
# index of values that get stored
# in each edge
PPRIOR = 0
NCOUNT = 1
WACCUMVALUE = 2
QMEANVALUE = 3


def compress_indices(tensor):
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
    def __init__(self, statemachine, model, color=WHITE):
        """
        :param StateMachine statemachine: is a state machine that
        translates from tensor indices to fen/uci notation and
        keeps track of state; also translates back to tensor notation
        :param model: trained keras model that does inference on
        position tensors s and returns move priors pi and position
        evaluation z
        :param np.array position: game state that is represented by
        this node in tensor notation
        :param int color: -1 for black, 1 for white
        """

        self.color = color
        self.statemachine = statemachine
        self.model = model

        if statemachine.has_ended():
            # game over: this is a leaf node
            # which represents a decisive state
            # of the game
            self.endposition = True

            result = statemachine.get_result()
            self.evaluation = result * self.color * AMPLIFY_RESULT
            self.children = []

        else: 
            # game still running
            self.endposition = False

            position = statemachine.get_position() 

            # TODO: do we actually need to store the position?
            self.position_shape = position.shape
            self.position = compress_indices(position)

            self.move_shape = statemachine.move_shape
            self.legal_move_indices = statemachine.get_legal_moves()

            num_of_legal_moves = len(self.legal_move_indices[0])
            self.children = [None] * num_of_legal_moves 

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
                         + " was average number of move possibilities per move"

        return tree_string + metric_string

    def _print_tree(self, level):
        l_shape = chr(9492)
        t_shape = chr(9500)
        i_shape = chr(9474)
        minus_shape = chr(9472)

        if level > 1000:
            return (0, 0, 0, 1, level, 0), "..."
        if self.endposition:
            return (0, 0, 1, 0, level, 0), "end, {:0.3f}".format(self.evaluation)
        if not any(self.children):
            return (0, 1, 0, 0, level, len(self.children)), \
                   "leaf, {:0.3f}".format(self.evaluation)

        rep = []
        metrics = (1, 0, 0, 0, level, len(self.children))
        maxlevel = level
        for i, j in enumerate(self.children):
            if j:
                # pylint: disable=W0212
                child_metrics, string = j._print_tree(level + 1)
                metrics = [k + l for k, l in zip(metrics, child_metrics)]
                maxlevel = max(maxlevel, child_metrics[4])
                rep.append(str(i) + ": " + string)
                # pylint: enable=W0212

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

    def get_policy_tensor(self):
        """
        :return: tensor with distributions from
        alpha zero tree search, that is the number of
        rollouts for each move divided by total number
        of rollouts in one move tensor
        :rtype: np.array
        """
        num_of_rollouts = self.edges[:, NCOUNT].sum()
        num_of_rollouts = max(1, num_of_rollouts)
        policy_weights = self.edges[:, NCOUNT] / num_of_rollouts
        policy_tensor = np.zeros(self.move_shape, EDGE_DTYPE)
        policy_tensor[self.legal_move_indices] = policy_weights
        return policy_tensor

    def get_move(self):
        """
        :return: best move according to current state of
        tree search in fen notation
        :rtype: str
        """
        i = np.argmax(self.edges[:, NCOUNT])
        i = self._legal_to_total_index(i)
        return self.statemachine.move_index_to_fen(i)

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

        move_idx = self._legal_to_total_index(i)
        self.statemachine.idx_move(move_idx)

        evaluation = 0

        if next_node is None:
            # terminate recursion
            # on leaf expansion 
            leaf = AztsNode(self.statemachine,
                            self.model,
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

        if level == 0:
            # after rollout is finished and
            # all values have been propagated
            # up, we are back at level 0 and
            # reset the state machine to this
            # game state.
            self.statemachine.reset_to_actual_game()

        return evaluation

    def _index_of_best_move(self):
        """
        :return: index of best move
        :rtype: int
        """

        U = self.edges[:, PPRIOR] / (self.edges[:, NCOUNT] + 1)
        U *= EXPLORATION
        Q = self.edges[:, QMEANVALUE]
        best_move_index = (Q + U).argmax()

        return best_move_index

    def _legal_to_total_index(self, index):
        """
        :param int index: index of a move in legal moves list
        :return: index of same move in move tensor notation
        :rtype: tuple
        """
        return tuple(np.array(self.legal_move_indices).T[index])

    def _position_as_tensor(self):
        """
        :return: tensor notation of current position
        :rtype: np.array
        """
        tensor = np.zeros(self.position_shape, dtype=POS_DTYPE)
        tensor[self.position] = 1
        return tensor

    def get_position(self):
        """
        :return: current position in tensor notation
        :rtype: np.array
        """
        board = np.zeros(self.position_shape, dtype=POS_DTYPE)
        board[self.position] = 1
        return board

    # pylint: enable=C0326

if __name__ == "__main__":
    statemachine = state_machine.StateMachine()
    mock_model = mock_model.MockModel()
    node = AztsNode(statemachine, mock_model)
    for i in range(25):
        node.rollout()

    print(node)
# pylint: enable=W0621
