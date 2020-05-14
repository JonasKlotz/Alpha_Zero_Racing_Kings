'''
mcts module containing classes
for the alpha zero tree search.
the mcts module is the entry
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
        self.move_shape = legal_moves.shape
        self.legal_move_indices = np.where(legal_moves == 1)
        num_of_legal_moves = len(self.legal_move_indices[0])
        self.children = [None] * num_of_legal_moves

        if num_of_legal_moves == 0:
            # reached end of game
            self.evaluation = translator.get_result(position)
            self.endposition = True

        else:
            # expansion of node
            policy, self.evaluation = model.inference(position)

            # initialise tensor to hold
            # 5 values per edge
            entries_per_edge = 5
            self.edges = np.zeros(
                    (*legal_moves[self.legal_move_indices].shape,
                        entries_per_edge))

            # only store prior values of legal moves
            self.edges[:, PPRIOR] = policy[self.legal_move_indices]

            # for every legal move,
            # store list index which refers to
            # corresponding child node
            self.edges[:, CHILDNODE] = np.arange(num_of_legal_moves)

    def __str__(self):
        return self._print_tree()

    def _print_tree(self, level = 0):
        l_shape = chr(9492)
        t_shape = chr(9500)
        i_shape = chr(9474)
        minus_shape = chr(9472)

        if level > 25:
            return "..."
        elif self.endposition:
            return "end, {:0.3f}".format(self.evaluation)
        elif not any(self.children):
            return "leaf, {:0.3f}".format(self.evaluation)

        rep = []
        for i, j in enumerate(self.children):
            if j:
                rep.append(str(i) + ": " + j._print_tree(level + 1))

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
        return repstr 

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
            # * expand position to tensor
            # * set translator to this position
            # * expand node index to move tensor
            # * set move to translator
            # * get resulting position
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

        U = self.edges[:,PPRIOR] / (self.edges[:,NCOUNT] + 1)
        Q = self.edges[:,QMEANVALUE]
        best_move_index = (Q + U).argmax()

        return best_move_index

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
