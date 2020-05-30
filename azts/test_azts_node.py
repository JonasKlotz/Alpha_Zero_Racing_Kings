# pylint: disable=E0401
# pylint: disable=E0602
# pylint: disable=C0111
# pylint: disable=W0621
import numpy as np
import pytest
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
STALE_MATE = "8/8/8/8/8/8/R7/5K1k b - - 10 20"
WIN_STATE = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
NUM_OF_LEGAL_START_MOVES = 21
MOVE_TENSOR_SHAPE = (8, 8, 64) 
LEGAL_START_MOVE_INDICES = np.array([[7, 6, 0], [7, 6, 7], \
        [5, 6, 7], [5, 6, 23], [4, 7, 62]])

@pytest.fixture
def initial_node():
    model = mock_model.MockModel()
    statemachine = state_machine.StateMachine()
    node = azts_node.AztsNode(statemachine, model)
    return node

@pytest.fixture
def rollout_node():
    model = mock_model.MockModel()
    # testing: model responds with all ones
    when(model).inference(...).thenReturn((np.ones(MOVE_TENSOR_SHAPE), 1))
    statemachine = state_machine.StateMachine()
    node = azts_node.AztsNode(statemachine, model)
    for i in range(10):
        node.rollout()
    return node

@pytest.fixture
def stale_node():
    model = mock_model.MockModel()
    statemachine = state_machine.StateMachine()
    statemachine.set_to_fen_state(STALE_MATE)
    node = azts_node.AztsNode(statemachine, model, BLACK)
    return node

@pytest.fixture
def win_node():
    model = mock_model.MockModel()
    statemachine = state_machine.StateMachine()
    statemachine.set_to_fen_state(WIN_STATE)
    node = azts_node.AztsNode(statemachine, model, BLACK)
    return node

@pytest.fixture
def lose_node():
    model = mock_model.MockModel()
    statemachine = state_machine.StateMachine()
    statemachine.set_to_fen_state(WIN_STATE)
    node = azts_node.AztsNode(statemachine, model, WHITE)
    return node

def test_compress_indices():
    compressed = azts_node.compress_indices(TEST_TENSOR)
    assert np.array(compressed).sum() == np.array(TEST_INDICES).sum()

def test_actual_start_position(initial_node):
    assert initial_node.statemachine.actual_game.board.fen() == FIRST_STATE

def test_rollout_start_position(initial_node):
    assert initial_node.statemachine.rollout_game.board.fen() == FIRST_STATE

def test_start_is_not_endpostion(initial_node):
    assert initial_node.endposition is False 

def test_stats_num_of_nodes_on_start(initial_node):
    assert initial_node.get_move_statistics()["tree"]["number of nodes"] == 1

def test_correct_num_of_children_on_start(initial_node):
    assert len(initial_node.children) == NUM_OF_LEGAL_START_MOVES

def test_get_policy_tensor_empty_on_start(initial_node):
    assert initial_node.get_policy_tensor().sum() == 0

def test_get_policy_tensor_correct_shape(initial_node):
    assert initial_node.get_policy_tensor().shape == MOVE_TENSOR_SHAPE 

def test_correct_start_moves(initial_node):
    for i in LEGAL_START_MOVE_INDICES:
        assert i in np.array(initial_node.legal_move_indices).T

def test_correct_print_function(initial_node):
    assert initial_node.__str__()[0:4] == "leaf"

def test_correct_number_of_nodes_in_tree(rollout_node):
    tree = rollout_node.__str__()
    idx = tree.find(" nodes in total")
    num_of_nodes = int(tree[idx-2:idx])
    assert num_of_nodes == 11

def test_move_distribution_adds_up_to_one(rollout_node):
    assert np.isclose(rollout_node.get_move_distribution().sum(), 1)

def test_move_stats_add_to_one(rollout_node):
    stats = rollout_node.get_move_statistics()["move distribution"]
    accum = 0
    keys = [i for i in stats.keys() if "rating" in i]
    for i in keys:
        accum += stats[i]
    assert np.isclose(accum, 1)

def test_move_stats_small_rest_in_low_heat(rollout_node):
    rest = rollout_node.get_move_statistics(0.001)["move distribution"]["rest rating"]
    assert rest < 0.1

def test_move_distribution_prioritizes_correct_move(rollout_node):
    best_move_from_distribution = np.argmax(rollout_node.get_move_distribution())
    best_move_from_edges = np.argmax(rollout_node.edges[:, 1])
    assert best_move_from_distribution == best_move_from_edges 

def test_stats_correct_type(rollout_node):
    assert isinstance(rollout_node.get_move_statistics(), dict)

def test_stats_num_of_nodes(rollout_node):
    assert rollout_node.get_move_statistics()["tree"]["number of nodes"] == 11

def test_stats_tree_overflow(rollout_node):
    assert rollout_node.get_move_statistics()["tree"]["tree overflow"] is False

def test_actual_game_position_is_still_set_after_rollout(rollout_node):
    assert rollout_node.statemachine.actual_game.board.fen() == FIRST_STATE

def test_rollout_game_position_is_reset_after_rollout(rollout_node):
    assert rollout_node.statemachine.rollout_game.board.fen() == FIRST_STATE

def test_evaluation_is_set_to_model_inference(rollout_node):
    assert rollout_node.evaluation == 1

def test_get_policy_tensor_normalized(rollout_node):
    assert rollout_node.get_policy_tensor().sum() == 1

def test_correct_print_function_for_more_nodes(rollout_node):
    assert rollout_node.__str__()[0:4] == "node"

def test_actual_start_position_after_rollout(rollout_node):
    assert rollout_node.statemachine.actual_game.board.fen() == FIRST_STATE

def test_rollout_start_position_after_rollout(rollout_node):
    assert rollout_node.statemachine.rollout_game.board.fen() == FIRST_STATE

def test_node_is_evaluated_to_one_by_mock_model(rollout_node):
    assert rollout_node.evaluation == 1 

def test_no_children_in_end_state(stale_node):
    assert stale_node.children == []

def test_end_state_is_endposition(stale_node):
    assert stale_node.endposition is True

def test_stale_mate_evaluates_to_zero(stale_node):
    assert stale_node.rollout() == 0

def test_actual_start_position_after_set_board(stale_node):
    assert stale_node.statemachine.actual_game.board.fen() == STALE_MATE

def test_rollout_start_position_after_set_board(stale_node):
    assert stale_node.statemachine.rollout_game.board.fen() == STALE_MATE 

def test_win_evaluation_black(win_node):
    assert win_node.evaluation == AMPLIFY_RESULT

def test_lose_evaluation_white(lose_node):
    assert lose_node.evaluation == -AMPLIFY_RESULT

# pylint: enable=E0401
# pylint: enable=E0602
# pylint: enable=C0111
# pylint: enable=W0621
