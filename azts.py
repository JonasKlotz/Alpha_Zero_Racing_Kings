'''
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
'''
import numpy as np
import StateMachine as sm

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
EDGE_DTYPE = np.float16
IDX_DTYPE = np.uint16

SELFPLAY = True

WHITE = 1
BLACK = -1
SWAP = -1

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


class Azts():
    '''
    Azts represents the
    alpha zero search tree.
    '''
    def __init__(self, 
            state_machine,
            model,
            position,
            color,
            sims_per_move=10):

        self.color = color

        self.state_machine = state_machine
        self.model = model
        self.root = Node(state_machine,
                model,
                state_machine.get_actual_position(),
                color)
        self.sims_per_move = sims_per_move

    def _tree_search(self):
        for i in range(self.sims_per_move):
            self.root.rollout()
            self.state_machine.reset_to_actual_game()

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
    def __init__(self, state_machine, model, position, color = WHITE):

        self.color = color

        self.state_machine = state_machine
        self.model = model
        self.evaluation = 0
        self.endposition = False

        self.position_shape = position.shape
        self.position = self._compress_indices(position)

        legal_moves = None
        if SELFPLAY:
            print(f"collecting legal moves for new node ...")
            legal_moves = state_machine.get_legal_moves() 
        else:
            legal_moves = state_machine.get_legal_moves(position)

        self.move_shape = legal_moves.shape
        self.legal_move_indices = self._compress_indices(legal_moves)
        num_of_legal_moves = len(self.legal_move_indices[0])
        self.children = [None] * num_of_legal_moves

        if num_of_legal_moves == 0:
            # reached end of game
            result = state_machine.get_rollout_result()
            self.evaluation = result * self.color
            self.endposition = True

        else:
            # expansion of node
            policy, self.evaluation = model.inference(position)

            # initialise tensor to hold
            # 5 values per edge
            entries_per_edge = 4
            self.edges = np.zeros(
                    (*legal_moves[self.legal_move_indices].shape,
                        entries_per_edge),
                    dtype = EDGE_DTYPE)

            # only store prior values of legal moves
            self.edges[:, PPRIOR] = policy[self.legal_move_indices] 

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

    def _compress_indices(self, tensor):
        '''
        tensor is a positional tensor with
        entries 0 and 1. compress indices
        extracts the indices of entries 1 in
        tensor and converts them into reduced
        data type
        '''
        indices = np.where(tensor == 1)
        compressed = []
        for i in indices:
            compressed.append(i.astype(IDX_DTYPE))

        return tuple(compressed)

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
        nextnode = self.children[i]

        newposition = None

        move = self._move_as_tensor(i)
        if SELFPLAY:
            # in self-play, we let the state
            # machine keep track of every move
            # to catch move-repetitions as draws
            # so that we get a realistic data set
            newposition = self.state_machine.rollout_tensor_move(move)

        evaluation = 0

        if nextnode is None:
            # terminate recursion
            # on leaf expansion
            if not SELFPLAY:
                # if not in self-play, we do not use
                # the state machine, because all
                # the conversions tensor-fen are
                # quite expensive and evaluation
                # depends on the model anyways
                oldposition = self._position_as_tensor() 
                newposition = self.state_machine.get_new_position(
                        oldposition, move)

            leaf = Node(self.state_machine,
                    self.model, 
                    newposition,
                    self.color)
            evaluation = leaf.evaluation
            self.children[i] = leaf

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
        4 values P, N, W, Q
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

    def _position_as_tensor(self):
        '''
        this node represents a game
        position, which is stored in
        compressed form.
        _position_as_tensor returns
        the decompressed tensor notation
        of that position.
        '''
        tensor = np.zeros(self.position_shape, dtype = POS_DTYPE)
        tensor[self.position] = 1
        return tensor

    def _move_as_tensor(self, move_index):
        '''
        takes a move index which refers
        to an edge in edges and converts
        this single move to tensor notation.
        tensor is of data type POS_DTYPE,
        because entries are only zeros and
        one 1 for the selected move.
        '''
        tensor = np.zeros(self.move_shape, dtype = POS_DTYPE)
        i = self._legal_to_total_index(move_index)
        tensor[i] = 1
        return tensor

    def _edges_as_tensor(self):
        '''
        return move recommendation
        distribution over all legal
        moves in one tensor according
        to tensor notation of moves.
        '''
        tensor = np.zeros(self.move_shape, dtype = MOVE_DTYPE)
        #TODO: actually put values there.
        return tensor


    # pylint: enable=C0326


def set_up():
    state_machine = sm.StateMachine()
    model = MockModel()
    node = Node(state_machine,
            model,
            state_machine.get_actual_position())

    return state_machine, model, node



if __name__ == "__main__":
    state_machine, model, node = set_up()
    for i in range(10):
        node.rollout()
        state_machine.reset_to_actual_game()

    print(node)
