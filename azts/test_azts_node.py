import numpy as np
from mockito import when, mock, unstub

from azts import azts_node
from azts import state_machine
from azts import mock_model

# pylint: disable=C0116
TEST_INDICES = (np.array([1, 4, 4, 2, 1]), \
        np.array([1, 0, 0, 1, 3]), \
        np.array([1, 0, 3, 1, 4]))

TEST_TENSOR = np.zeros((5, 5, 5))
TEST_TENSOR[TEST_INDICES] = 1

def test_compress_indices():
    compressed = azts_node.compress_indices(TEST_TENSOR)
    assert np.array(compressed).sum() == np.array(TEST_INDICES).sum() 

model = mock_model.MockModel()
statemachine = state_machine.StateMachine()
node = azts_node.AztsNode(statemachine, model,\
        statemachine.get_actual_position())

NUM_OF_LEGAL_START_MOVES = 21

def test_correct_num_of_children_on_start():
    assert len(node.children) == NUM_OF_LEGAL_START_MOVES

def test_get_policy_tensor_empty_on_start():
    assert node.get_policy_tensor().sum() == 0

MOVE_TENSOR_SHAPE = (8, 8, 64)
def test_get_policy_tensor_correct_shape():
    assert node.get_policy_tensor().shape == MOVE_TENSOR_SHAPE

LEGAL_START_MOVE_INDICES = np.array([[7, 6, 0], [7, 6, 7], \
        [5, 6, 7], [5, 6, 23], [4, 7, 62]])

def test_correct_start_moves():
    for i in LEGAL_START_MOVE_INDICES:
        assert i in np.array(node.legal_move_indices).T 

def test_correct_print_function():
    assert node.__str__()[0:4] == "leaf"

# testing: model responds with all ones
when(model).inference(...).thenReturn((np.ones(MOVE_TENSOR_SHAPE), 1))

node_rollout = azts_node.AztsNode(statemachine, model, \
        statemachine.get_actual_position())

for i in range(10):
    node_rollout.rollout()

def test_get_policy_tensor_normalized():
    assert node_rollout.get_policy_tensor().sum() == 1 

def test_correct_print_function_for_more_nodes():
    assert node_rollout.__str__()[0:4] == "node"

# pylint: enable=C0116
