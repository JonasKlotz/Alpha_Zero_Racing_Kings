'''
tests for classes mcts
and node
'''
from mockito import when, mock, unstub
import numpy as np
import mcts


class Translator():
    '''
    mock translator
    '''
    def get_legal_moves(self, position):
        return 0


class Model():
    '''
    mock model
    '''
    def inference(self, position):
        return 0, 0


def random_start(move_dim, pos_dim):
    legals = np.random.rand(*move_dim)
    legals[legals > 0.5] = 1
    legals[legals < 0.6] = 0

    pos = np.random.rand(*pos_dim)
    pos[pos > 0.5] = 1
    pos[pos < 0.6] = 0

    when(MODEL).inference(...).thenReturn((np.ones((move_dim)), 0.5))
    when(TRANS).get_legal_moves(...).thenReturn(legals)

    return legals, pos


MODEL = Model()
TRANS = Translator()
MOV, POS = random_start((3, 3, 3), (5, 2, 4)) 

def test_node_init_legal_moves():
    '''
    tests if edges represent
    legal moves correctly
    '''
    node = mcts.Node(TRANS, MODEL, POS) 
    for i, j in zip(node.legal_move_indices, np.where(MOV == 1)):
        assert np.all(i == j)

MOV, POS = random_start((5, 2, 4), (3, 3, 3))

def test_node_init_legal_moves_diff_dims():
    '''
    tests if edges represent
    legal moves correctly
    for new set of dimensions
    '''
    node = mcts.Node(TRANS, MODEL, POS)
    for i, j in zip(node.legal_move_indices, np.where(MOV == 1)):
        assert np.all(i == j)
