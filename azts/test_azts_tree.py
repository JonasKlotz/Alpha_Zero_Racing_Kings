# pylint: disable=C0116
import numpy as np
from mockito import when, mock, unstub
import pytest

from azts import azts_tree
from azts import azts_node
from azts import state_machine
from azts import mock_model

from azts.config import BLACK, AMPLIFY_RESULT, WHITE

@pytest.fixture
def tree_white_start():
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = azts_tree.AztsTree(statemachine, \
            model, WHITE, 10)
    return tree

@pytest.fixture
def tree_white_other_turn():
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = azts_tree.AztsTree(statemachine, \
            model, WHITE, 10)
    tree.make_move()
    return tree

@pytest.fixture
def tree_stale_mate():
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = azts_tree.AztsTree(statemachine, \
            model, BLACK, 10)
    stale_mate = "8/8/8/8/8/8/R7/5K1k b - - 10 20"
    tree.set_to_fen_state(stale_mate)
    return tree

@pytest.fixture
def tree_won(): 
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = azts_tree.AztsTree(statemachine, \
            model, BLACK, 10)
    win_position = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    tree.set_to_fen_state(win_position)
    return tree

def test_tree_and_root_share_statemachine(\
        tree_white_start):
    assert tree_white_start.statemachine \
        is tree_white_start.root.statemachine

def test_policy_tensor_correct_dimensions(\
        tree_white_start):
    assert tree_white_start.get_policy_tensor().shape == (8, 8, 64)

def test_dont_receive_moves_in_own_turn(\
        tree_white_start):
    with pytest.raises(Exception):
        tree_white_start.receive_move("a2a3") 

def test_root_print_function(\
        tree_white_start):
    assert type(tree_white_start.root.__str__()) == str 

def test_root_shares_statemachine_with_tree(\
        tree_white_start):
    assert tree_white_start.statemachine \
        is tree_white_start.root.statemachine

def test_legal_move_from_start(\
        tree_white_start):
    legal_moves = \
        tree_white_start.statemachine.actual_game.get_moves_observation()
    move = tree_white_start.make_move()
    assert move in legal_moves

def test_game_start_not_over(\
        tree_white_start):
    assert tree_white_start.game_over() is False

def test_game_state_running(\
        tree_white_start):
    assert tree_white_start.game_state() == "running"

def test_game_start_result_zero(\
        tree_white_start):
    assert tree_white_start.game_result() == 0

def test_no_move_in_other_turn(tree_white_other_turn):
    with pytest.raises(Exception):
        tree_white_other_turn.make_move()

def test_dont_receive_illegal_move(tree_white_other_turn):
    illegal_move = "a1a8"
    with pytest.raises(Exception):
        tree_white_other_turn.receive_move(illegal_move)

def test_game_is_actually_reset(\
        tree_white_other_turn):
    new_state = "r6R/8/8/8/8/k6K/2bnNB2/qrbnNBRQ w - - 0 1"
    tree_white_other_turn.set_to_fen_state(new_state)
    assert \
        tree_white_other_turn.statemachine.actual_game.board.fen() \
        == new_state 

def test_dont_receive_move_if_reset_to_white_turn(\
        tree_white_other_turn):
    new_state = "r6R/8/8/8/8/k6K/2bnNB2/qrbnNBRQ w - - 0 1"
    tree_white_other_turn.set_to_fen_state(new_state)
    with pytest.raises(Exception):
        tree_white_other_turn.receive_move("a8a7")

def test_exception_on_stale_mate(tree_stale_mate):
    with pytest.raises(Exception):
        tree_stale_mate.make_move()

def test_game_state_on_stale_mate(tree_stale_mate):
    assert tree_stale_mate.game_state() == "draw by stale mate"

def test_stalemate_game_over(\
        tree_stale_mate):
    assert tree_stale_mate.game_over()

def test_stalemate_result_zero(\
        tree_stale_mate):
    assert tree_stale_mate.game_result() == 0

def test_won_game_over(tree_won):
    assert tree_won.game_over()

def test_won_registered(tree_won):
    assert tree_won.statemachine.actual_game.board.is_variant_end()

def test_won_result(tree_won):
    assert tree_won.game_result() == BLACK

def test_won_game_state(tree_won):
    assert tree_won.game_state() == "black won"

# pylint: enable=C0116
