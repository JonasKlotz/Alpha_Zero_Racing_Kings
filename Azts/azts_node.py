"""
Node of the Alpha Zero Tree Search.
Each node represents a game state
and holds all possible moves as
edges to other states. Edges keep
track of the variables of the
algorithm: N, W, Q, P.
"""
import numpy as np

from Azts import state_machine
from Azts import mock_model 
from Azts.config import WHITE, POS_DTYPE,\
        EDGE_DTYPE, IDX_DTYPE, \
        EXPLORATION, ROLLOUT_PAYOFFS
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
    def __init__(self, \
            statemachine, \
            model, \
            color=WHITE, \
            exploration=EXPLORATION, \
            payoffs=ROLLOUT_PAYOFFS):
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
        self.payoffs = payoffs
        self.exploration = exploration

        position = statemachine.get_position() 
        self.position_shape = position.shape
        self.position = compress_indices(position)
        self.move_shape = statemachine.move_shape

        if statemachine.game_over():
            # game over: this is a leaf node
            # which represents a decisive state
            # of the game
            self.endposition = True

            state = statemachine.get_state()
            self.evaluation = self.payoffs[self.color][state]
            self.children = []
            self.edges = None


        else: 
            # game still running
            self.endposition = False 

            self.legal_move_indices = statemachine.get_legal_moves()

            num_of_legal_moves = len(self.legal_move_indices[0])
            self.children = [None] * num_of_legal_moves 

            # expansion of node
            policy, self.evaluation = model.inference(position)

            self.evaluation = self.evaluation if self.color == WHITE \
                    else -self.evaluation

            # initialise tensor to hold
            # 4 values per edge
            entries_per_edge = 4
            self.edges = np.zeros(
                (num_of_legal_moves, entries_per_edge),
                dtype=EDGE_DTYPE)

            # only store prior values of legal moves
            edges_sum = policy[self.legal_move_indices].sum()
            self.edges[:, PPRIOR] = policy[self.legal_move_indices] / edges_sum
            


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
                         + " was average number of move possibilities per move\n\n" \
                         + "Move statistics:\n"
        move_stats = self._get_distribution_statistics()
        for i in move_stats.keys():
            filler = {"move": "\t\t\t", "score": "\t\t\t\t", "probability": "\t", \
                    "prior": "\t\t\t\t\t", "visits": "\t\t\t\t\t\t"}
            select = i.split("_")[1]
            distr_metric = f"\t{i}:{filler[select]}{str(move_stats[i])[0:5]}\n"
            metric_string += distr_metric 

        metric_string += "\n\nNote that the scores change dynamically\n" \
                + "during rollouts and do not determine\n" \
                + "the distribution; they rather determine\n" \
                + "the first move for the next rollout."

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
        edge_scores = self._edge_scores()
        for i, j in enumerate(self.children):
            if j:
                # pylint: disable=W0212
                child_metrics, string = j._print_tree(level + 1)
                score = str(edge_scores[i])[0:5]
                metrics = [k + l for k, l in zip(metrics, child_metrics)]
                maxlevel = max(maxlevel, child_metrics[4])
                rep.append(f"{i}: scored {score}: {string}")
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

    def _get_tree_statistics(self):
        '''
        get information about the current state of the rollout tree
        :return dict: dictionary containing several kpis
        '''
        metrics, _ = self._print_tree(0)

        num_of_nodes = metrics[0] + metrics[1] + metrics[2]
        avg_legal_moves = metrics[5] / num_of_nodes
        tree_overflow = metrics[3] != 0
        
        stats = {"tree_overflow": tree_overflow, \
                "number_of_nodes": num_of_nodes, \
                "leaf_nodes": metrics[1], \
                "end_positions": metrics[2], \
                "normal_tree_nodes": metrics[0], \
                "maximal_tree_depth": metrics[4], \
                "avg_num_of_legal_moves_per_pos": avg_legal_moves, \
                "num_of_legal_moves": len(self.children)}
        
        return stats

    def _get_distribution_statistics(self, heat = 1):
        '''
        gather information about the move distribution
        and return it as dictionary
        '''

        move_distribution = self.get_move_distribution(heat) 
        scores = self._edge_scores()
        stats = {}

        for i in ["first", "second", "third", "fourth"]:
            j = i + "_probability"
            k = i + "_move"
            l = i + "_score"
            m = i + "_prior"
            n = i + "_visits"
            select = move_distribution.argmax()
            stats[j] = float(move_distribution[select])
            move_index = self._legal_to_total_index(select)
            move = self.statemachine.move_index_to_fen(move_index)
            stats[k] = move
            move_distribution[select] = 0
            stats[l] = float(scores[select])
            stats[m] = float(self.edges[select][PPRIOR])
            stats[n] = float(self.edges[select][NCOUNT])

        stats["rest_probability"] = float(move_distribution.sum())
        return stats
    
    def get_move_statistics(self, heat = 1):
        stats = {}
        stats["tree"] = self._get_tree_statistics()
        stats["move_distribution"] = self._get_distribution_statistics(heat) 

        return stats


    def get_policy_tensor(self):
        """
        :return: tensor with distributions from
        alpha zero tree search, that is the number of
        rollouts for each move divided by total number
        of rollouts in one move tensor
        :rtype: np.array
        """
        policy_tensor = np.zeros(self.move_shape, EDGE_DTYPE)
        if self.edges is not None: 
            num_of_rollouts = self.edges[:, NCOUNT].sum()
            num_of_rollouts = max(1, num_of_rollouts)
            policy_weights = self.edges[:, NCOUNT] / num_of_rollouts
            policy_tensor = np.zeros(self.move_shape, EDGE_DTYPE)
            policy_tensor[self.legal_move_indices] = policy_weights

        return policy_tensor

    def get_move_distribution(self, heat=1):
        '''
        calculates the current move distribution from
        which the next move is being sampled. Every legal
        move is being assigned a probability which is determined
        by the simulations of that move; all move probabilities
        add up to 1.
        :param float heat: control explorative behaviour:
        heat > 1 more exploration (less deterministic)
        0 < heat < 1 more exploitation
        :return np.array: distribution of moves; indices in this
        array correspond to the indices in self.edges and need
        to be translated to actual moves with _legal_to_total_index
        '''
        distribution = self.edges[:, NCOUNT] / max(self.edges[:, NCOUNT].sum(), 1)
        if heat != 1:
            heat = max(heat, 0.0001)
            distribution = np.power(distribution, 1 / heat)
            distribution /= distribution.sum()

        return distribution

    def get_move(self, heat=1):
        """
        :return: best move according to current state of
        tree search in fen notation and node representing
        that move
        :rtype: str, AztsNode
        """
        distribution = self.get_move_distribution(heat)

        order = distribution.argsort()
        draw = np.random.rand(1)[0]
        category = 0

        # going through the distribution from smallest to
        # largest value
        for i in order:
            category += distribution[i]
            if draw < category:
                # select this move.
                move = self._legal_to_total_index(i)
                return self.statemachine.move_index_to_fen(move), \
                        self.children[i]

        # something went wrong: return best move 
        i = np.argmax(self.edges[:, NCOUNT])
        i = self._legal_to_total_index(i)
        return self.statemachine.move_index_to_fen(i), \
                self.children[i]

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
            leaf = AztsNode(statemachine=self.statemachine, \
                            model=self.model, \
                            color=self.color, \
                            exploration=self.exploration, \
                            payoffs=self.payoffs) 

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

    def _edge_scores(self):
        '''
        from the four metrics in each edge, calculate
        a score for each edge. highest score is being
        selected in rollouts.
        :return np.array: list of scores whos indices
        are aligned to self.edges
        '''
        U = self.edges[:, PPRIOR] / (self.edges[:, NCOUNT] + 1)
        U *= self.exploration
        Q = self.edges[:, QMEANVALUE]

        return Q + U

    def _index_of_best_move(self):
        """
        :return: index of best move
        :rtype: int
        """ 
        best_move_index = self._edge_scores().argmax()
        return best_move_index

    def _legal_to_total_index(self, index):
        """
        :param int index: index of a move in legal moves list
        :return: index of same move in move tensor notation
        :rtype: tuple
        """
        return tuple(np.array(self.legal_move_indices).T[index])

    def select_node_with_move(self, move):
        move_idx = self.statemachine.uci_to_move_idx(move)
        # compare which legal move index entries are all
        # equal to the move index of received move
        i = np.where(np.all(np.array(self.legal_move_indices).T \
                == move_idx, axis=1) == True)[0][0]
        if i is None:
            raise Exception(f"Received move {move} is not in children")
        return self.children[i]

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
