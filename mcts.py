'''
mcts module containing classes
for the alpha zero tree search.
the mcts module is the entry
point and the actual tree is
built with node objects defined
in the node module.  
'''
import numpy as np

# dimensions of mock object tensors
DIMENSIONS = (3, 3, 3)

# index of values that get stored
# in each edge
PPRIOR = 0
NCOUNT = 1
WACCUMVALUE = 2
QMEANVALUE = 3
CHILDNODE = 4



class MockTranslator():
    def get_legal_moves(self, position):
        legal_moves = np.random.rand(*DIMENSIONS)
        legal_moves[legal_moves > 0.5] = 1
        legal_moves[legal_moves < 0.6] = 0
        return legal_moves


class MockModel():
    def inference(self, position):
        policy = np.random.rand(*DIMENSIONS)
        evaluation = np.random.rand(1)[0]

        return (policy, evaluation)


class Mcts():
    '''
    Mcts represents the
    alpha zero search tree.
    '''
    def __init__(self, translator, model, position, sims_per_move=10):
        self.translator = translator
        self.model = model
        self.root = Node(translator, model, position)
        self.sims_per_move = sims_per_move

    def _tree_search(self):
        for i in range(self.sims_per_move):
            self.root.rollout()

    def _set_root_to(position):
        pass




class Node():
    '''
    Node of the Alpha Zero Tree Search.
    Each node represents a game state
    and holds all possible moves as
    edges to other states. Edges keep
    track of the variables of the
    algorithm: N, W, Q, P.
    '''
    # pylint: disable=C0326
    def __init__(self, translator, model, position):

        self.translator = translator
        self.model = model
        self.evaluation = 0
        self.endposition = False

        self.position_shape = position.shape
        self.position = np.where(position == 1)

        legal_moves = translator.get_legal_moves(position)
        self.legal_move_indices = np.where(legal_moves == 1)
        num_of_legal_moves = len(self.legal_move_indices[0])
        self.children = [None] * num_of_legal_moves

        if num_of_legal_moves == 0:
            # reached end of game
            self.evaluation = translator.get_result(position)
            self.endposition = True

        else:
            # expanding leaf node
            policy, self.evaluation = model.inference(position)
            p_legal = policy * legal_moves

            # initialise tensor to hold
            # 5 values per edge
            entries_per_edge = 5
            self.edges = np.zeros((*legal_moves.shape, entries_per_edge))

            # store prior values from
            # p_legal in edges
            self.edges[:, :, :, PPRIOR] = p_legal

            # for every legal move,
            # store list index which refers to
            # corresponding child node
            self._indicate_legal_moves()

    def __str__(self):
        return self._print_tree()

    def _print_tree(self, level = 0):

        indent = "\t" * level
        representation = ""


        children = ""

        for child in self.children:
            if child:
                children += child._print_tree(level + 1)

        if level == 0:
            representation = "root\n"
        elif self.endposition:
            representation = indent + "end\n"
        elif children == "":
            representation = indent + "leaf\n"
        else:
            representation = indent + "node\n"

        representation += children 

        return representation

    def rollout(self):
        '''
        recursive traversal of game tree
        updates P, N, W, Q of all travelled
        edges
        '''
        if self.endposition:
            # terminate recursion
            # if end state is reached
            return self.evaluation

        i = self._index_of_best_move()
        j = int(self.edges[i][CHILDNODE])
        nextnode = self.children[j]
        evaluation = 0

        if nextnode is None:
            # terminate recursion
            # on leaf expansion
            # TODO: where do we get new position?
            newposition = self.translator.get_legal_moves(None)
            leaf = Node(self.translator, self.model, newposition)
            evaluation = leaf.evaluation
            self.children[j] = leaf

        else:
            # recursion
            evaluation = nextnode.rollout()

        # calculate edge stats
        count = self.edges[i][NCOUNT] + 1
        accum = self.edges[i][WACCUMVALUE] + evaluation

        # update edge stats
        self.edges[i][QMEANVALUE] = accum / count
        self.edges[i][NCOUNT] = count
        self.edges[i][WACCUMVALUE] = accum

        return evaluation

    def _index_of_best_move(self):
        '''
        returns the index of the best move
        which then can be used to access the
        5 values P, N, W, Q and childnode
        '''

        legal_moves = self._all_legal_moves()

        U = legal_moves[:,PPRIOR] / (legal_moves[:,NCOUNT] + 1)
        Q = legal_moves[:,QMEANVALUE]
        best_move = (Q + U).argmax()
        move_index = self._legal_to_total_index(best_move)

        return move_index

    def _all_legal_moves(self):
        '''
        returns only the legal moves
        '''
        return self.edges[self.legal_move_indices]

    def _indicate_legal_moves(self):
        '''
        initializes all legal moves
        with the index where the
        corresponding child node is
        stored in self.children
        '''
        for i, j in enumerate(zip(*self.legal_move_indices)):
            self.edges[(*j, CHILDNODE)] = i 

    def _legal_to_total_index(self, index):
        '''
        translates index to a move
        in the legal move selection
        to an index to the same move
        in all moves
        '''
        return tuple(np.array(self.legal_move_indices).T[index])    


    # pylint: enable=C0326


def set_up():
    trans = MockTranslator()
    model = MockModel()
    x = trans.get_legal_moves(None)
    node = Node(trans, model, x)
    return trans, model, x, node
