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
        p = np.random.rand(*DIMENSIONS)
        v = np.random.rand(1)[0]

        return (p, v)


class Mcts():
    def __init__(self, translator, model, sims_per_move=10):
        self.translator = translator
        self.modell = modell
        self.root = mcts_node(translator, model)
        self.sims_per_move = sims_per_move

    def get_move_distribution(self, position):
        for i in range(self.sims_per_move):
            simulation_run(position) 
        pass

    def simulation_run(self):
        pass



class Node():
    def __init__(self, translator, model, position):

        self.v = 0 
        self.endposition = False
        legal_moves = translator.get_legal_moves(position)
        self.legal_move_indices = np.where(legal_moves == 1)
        num_of_legal_moves = len(self.legal_move_indices[0])
        self.children = [None] * num_of_legal_moves

        if num_of_legal_moves == 0:
            # reached end of game
            self.v = translator.get_result(position)
            self.endposition = True

        else: 
            # expanding leaf node
            p, self.v = model.inference(position)
            p_legal = p * legal_moves 

            # initialise tensor to hold 
            # 5 values per edge
            entries_per_edge = 5
            self.edges = np.zeros((*legal_moves.shape, entries_per_edge))

            # store prior values from
            # p_legal in edges
            self.edges[:,:,:,PPRIOR] = p_legal 

            # for every legal move,
            # store list index which refers to
            # corresponding child node
            legal_edges = self.all_legal_moves()
            legal_edges[:,CHILDNODE] = np.arange(num_of_legal_moves) 


    def rollout(self):
        '''
        recursive traversal of game tree
        updates P, N, W, Q of all travelled
        edges
        '''
        if self.endposition:
            # terminate recursion
            # if end state is reached
            return self.v

        i = self.index_of_best_move()
        j = self.edges[i][CHILDNODE]
        nextnode = self.children[j]
        v = 0

        if nextnode == None:
            # terminate recursion
            # on leaf expansion
            # TODO: where do we get new position?
            newposition = Null
            leaf = Node(self.translator, self.model, newposition)
            v = leaf.v

        else:
            # recursion
            v = nextnode.rollout()

        # calculate edge stats
        count = self.edges[i][NCOUNT] + 1
        accum = self.edges[i][WACCUMVALUE] + v 

        # update edge stats
        self.edges[i][QMEANVALUE] = accum / count
        self.edges[i][NCOUNT] = count
        self.edges[i][WACCUMVALUE] = accum

        return v


    def index_of_best_move(self):
        '''
        returns the index of the best move
        which then can be used to access the 
        5 values P, N, W, Q and childnode
        '''

        legal_moves = self.all_legal_moves()

        U = legal_moves[:,PPRIOR] / (legal_moves[:,NCOUNT] + 1)
        Q = legal_moves[:,QMEANVALUE] 
        best_move = (Q + U).argmax()
        move_index = self.legal_to_total_index(best_move)

        return move_index


    def all_legal_moves(self):
        '''
        returns only the legal moves
        '''
        return self.edges[self.legal_move_indices]


    def legal_to_total_index(self, index):
        '''
        translates index to a move
        in the legal move selection
        to an index to the same move
        in all moves
        '''
        return tuple(np.array(self.legal_move_indices).T[index])















