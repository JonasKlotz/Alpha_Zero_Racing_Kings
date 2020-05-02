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


class mock_translator():
    def get_legal_moves(self, position):
        legal_moves = np.random.rand(*DIMENSIONS)
        legal_moves[legal_moves > 0.5] = 1
        legal_moves[legal_moves < 0.6] = 0
        return legal_moves


class mock_model():
    def inference(self):
        p = np.random.rand(*DIMENSIONS)
        v = np.random.rand(1)[0]

        return (p, v)


class mcts():
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



class mcts_node():
    def __init__(self, translator, model, position):

        legal_moves = translator.get_legal_moves(position)
        mask = legal_moves == 1
        num_of_legal_moves = mask.sum()
        self.children = [None] * num_of_legal_moves

        p, v = model.inference(position)
        p_legal = p * legal_moves 

        # initialise tensor to hold 
        # 5 values per edge
        entries_per_edge = 5
        edges = np.zeros((*legal_moves.shape, entries_per_edge))

        # store prior values from
        # p_legal in edges
        edges[:,:,:,PPRIOR] = p_legal 

        # for every legal move,
        # store list index which refers to
        # corresponding child node
        node_indices = *np.where(mask), np.ones(mask.sum(), dtype=np.int) * CHILDNODE 
        edges[node_indices] = np.arange(mask.sum())

        # save changes
        self.edges = edges
        self.mask = mask
        self.v = v

    def get_evaluation(self):
        return self.v

    def select_move(self):
        '''
        returns the index of the best move
        which then can be used to access the 
        5 values P, N, W, Q and childnode
        '''

        legal_moves = np.where(self.mask)
        edges = self.edges

        U = edges[legal_moves][:,PPRIOR] / (edges[legal_moves][:,NCOUNT] + 1)
        Q = edges[legal_moves][:,QMEANVALUE] 
        best_move = (Q + U).argmax()
        move_index = tuple(np.array(legal_moves).T[best_move])

        return move_index













