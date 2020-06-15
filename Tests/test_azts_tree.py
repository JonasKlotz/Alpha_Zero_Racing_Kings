# pylint: disable=E0401
# pylint: disable=E0602
# pylint: disable=C0111
# pylint: disable=W0621
import pytest

from Azts import azts_tree
from Azts import state_machine
from Azts import mock_model 
from Azts.config import WHITE, BLACK, \
        RUNNING, DRAW, DRAW_BY_STALE_MATE, \
        WHITE_WINS, BLACK_WINS

@pytest.fixture
def init_tree():
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = azts_tree.AztsTree(model=model, \
            color=WHITE, \
            rollouts_per_move=10)
    return tree

@pytest.fixture
def black_tree():
    statemachine = state_machine.StateMachine()
    model = mock_model.MockModel()
    tree = azts_tree.AztsTree(model=model, \
            color=BLACK, \
            rollouts_per_move=10)
    return tree 

@pytest.fixture
def tree_white_start(init_tree):
    return init_tree

@pytest.fixture
def tree_white_other_turn(init_tree):
    init_tree.make_move()
    return init_tree

@pytest.fixture
def tree_stale_mate(black_tree):
    stale_mate = "8/8/8/8/8/8/R7/5K1k b - - 10 20"
    black_tree.set_to_fen_state(stale_mate)
    return black_tree

@pytest.fixture
def tree_won(black_tree):
    win_position = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    black_tree.set_to_fen_state(win_position)
    return black_tree

@pytest.fixture
def suspension_draw(black_tree):
    suspended_draw = "7K/k7/7R/8/8/8/8/1R6 b - - 10 20"
    black_tree.set_to_fen_state(suspended_draw)
    return black_tree

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
    assert isinstance(tree_white_start.root.__str__(), str)

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
    assert tree_white_start.game_state() == RUNNING

def test_game_start_result_zero(\
        tree_white_start):
    assert tree_white_start.game_result() == RUNNING

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
    assert tree_stale_mate.game_state() == DRAW_BY_STALE_MATE

def test_stalemate_game_over(\
        tree_stale_mate):
    assert tree_stale_mate.game_over()

def test_stalemate_result_zero(\
        tree_stale_mate):
    assert tree_stale_mate.game_result() == DRAW

def test_won_game_over(tree_won):
    assert tree_won.game_over()

def test_won_registered(tree_won):
    assert tree_won.statemachine.actual_game.board.is_variant_end()

def test_won_result(tree_won):
    assert tree_won.game_result() == BLACK_WINS

def test_won_game_state(tree_won):
    assert tree_won.game_state() == BLACK_WINS

def test_suspension_draw_game_not_over(suspension_draw):
    assert suspension_draw.game_over() is False

def test_suspension_draw_game_state_running(suspension_draw):
    assert suspension_draw.game_state() == RUNNING

def test_suspension_draw_number_of_node_children(suspension_draw):
    assert len(suspension_draw.root.children) == 1
# pylint: enable=E0401
# pylint: enable=E0602
# pylint: enable=C0111
# pylint: enable=W0621
