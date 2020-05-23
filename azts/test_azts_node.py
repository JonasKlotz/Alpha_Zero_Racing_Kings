# pylint: disable=C0116
import numpy as np
from mockito import when, mock, unstub

from azts import azts_node
from azts import state_machine
from azts import mock_model

from azts.config import BLACK, AMPLIFY_RESULT, WHITE

TEST_INDICES = (np.array([1, 4, 4, 2, 1]), \
        np.array([1, 0, 0, 1, 3]), \
        np.array([1, 0, 3, 1, 4]))

TEST_TENSOR = np.zeros((5, 5, 5))
TEST_TENSOR[TEST_INDICES] = 1
FIRST_STATE = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"

def test_compress_indices():
    compressed = azts_node.compress_indices(TEST_TENSOR)
    assert np.array(compressed).sum() == np.array(TEST_INDICES).sum() 

model = mock_model.MockModel()
statemachine = state_machine.StateMachine()
node = azts_node.AztsNode(statemachine, model)

def test_actual_start_position():
    assert node.statemachine.actual_game.board.fen() == FIRST_STATE

def test_rollout_start_position():
    assert node.statemachine.rollout_game.board.fen() == FIRST_STATE

def test_start_is_not_endpostion():
    assert node.endposition == False

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

node_rollout = azts_node.AztsNode(statemachine, model)

for i in range(10):
    node_rollout.rollout()

def test_correct_number_of_nodes_in_tree():
    tree = node_rollout.__str__()
    idx = tree.find(" nodes in total")
    num_of_nodes = int(tree[idx-2:idx])
    assert num_of_nodes == 11 

def test_actual_game_position_is_still_set_after_rollout():
    assert node_rollout.statemachine.actual_game.board.fen() == FIRST_STATE

def test_rollout_game_position_is_reset_after_rollout():
    assert node_rollout.statemachine.rollout_game.board.fen() == FIRST_STATE

def test_evaluation_is_set_to_model_inference():
    assert node_rollout.evaluation == 1

def test_get_policy_tensor_normalized():
    assert node_rollout.get_policy_tensor().sum() == 1 

def test_correct_print_function_for_more_nodes():
    assert node_rollout.__str__()[0:4] == "node"

def test_actual_start_position_after_rollout():
    assert node_rollout.statemachine.actual_game.board.fen() == FIRST_STATE

def test_rollout_start_position_after_rollout():
    assert node_rollout.statemachine.rollout_game.board.fen() == FIRST_STATE

def test_node_is_evaluated_to_one_by_mock_model():
    assert node_rollout.evaluation == 1

STALE_MATE = "8/8/8/8/8/8/R7/5K1k b - - 10 20"

stale_statemachine = state_machine.StateMachine()
stale_statemachine.set_to_fen_state(STALE_MATE)
node_stale = azts_node.AztsNode(stale_statemachine, model, BLACK)

def test_no_children_in_end_state():
    assert node_stale.children == []

def test_end_state_is_endposition():
    assert node_stale.endposition == True

def test_stale_mate_evaluates_to_zero():
    assert node_stale.rollout() == 0

def test_actual_start_position_after_set_board():
    assert node_stale.statemachine.actual_game.board.fen() == STALE_MATE

def test_rollout_start_position_after_set_board():
    assert node_stale.statemachine.rollout_game.board.fen() == STALE_MATE

WIN_STATE = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
win_statemachine = state_machine.StateMachine()
win_statemachine.set_to_fen_state(WIN_STATE)
node_win = azts_node.AztsNode(win_statemachine, model, BLACK)

node_lose = azts_node.AztsNode(win_statemachine, model, WHITE)

def test_win_evaluation_black():
    assert node_win.evaluation == AMPLIFY_RESULT

def test_lose_evaluation_white():
    assert node_lose.evaluation == -AMPLIFY_RESULT

# pylint: enable=C0116
