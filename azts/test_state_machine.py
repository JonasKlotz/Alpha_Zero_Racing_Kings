# pylint: disable=C0116
import pytest

from azts import state_machine 
from azts.config import WHITE, BLACK 

statemachine = state_machine.StateMachine()

@pytest.fixture
def statemachine_init():
    return state_machine.StateMachine()

@pytest.fixture
def statemachine_stalemate():
    statemachine = state_machine.StateMachine()
    new_state = "8/8/8/8/8/8/R7/5K1k b - - 10 20" 
    statemachine.set_to_fen_state(new_state)
    return statemachine 

@pytest.fixture
def statemachine_win():
    statemachine = state_machine.StateMachine()
    win_state = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    statemachine.set_to_fen_state(win_state)
    return statemachine

@pytest.fixture
def statemachine_suspended():
    suspended_win = "7K/k7/8/8/8/8/7R/8 b - - 10 20"
    statemachine = state_machine.StateMachine()
    statemachine.set_to_fen_state(suspended_win)
    return statemachine

@pytest.fixture
def sm_noblacksuspense():
    no_black_suspense = "7k/K7/8/8/8/8/R7/8 w - - 10 20"
    statemachine = state_machine.StateMachine()
    statemachine.set_to_fen_state(no_black_suspense)
    return statemachine 

def test_two_games_are_independent(statemachine_init):
    assert statemachine_init.actual_game is not \
        statemachine_init.rollout_game

def test_correct_tensor_dimensions_for_position(statemachine_init): 
    assert statemachine_init.get_position().shape == (8, 8, 11)

def test_get_player_color_from_start(statemachine_init):
    assert statemachine_init.get_player_color() == WHITE

def test_start_game_has_not_ended(statemachine_init):
    assert statemachine_init.actual_game_over() == False

def test_get_legal_moves_right_number_of_moves(statemachine_init):
    num_of_legal_moves_from_start = 21
    moves = statemachine_init.get_legal_moves()
    assert len(moves[0]) == num_of_legal_moves_from_start

def test_get_rollout_result_from_start(statemachine_init):
    assert statemachine_init.get_result() == 0

def test_get_actual_result_from_start(statemachine_init):
    assert statemachine_init.get_actual_result() == 0

def test_exception_on_impossible_move(statemachine_init):
    impossible_move = "h1h8"
    with pytest.raises(ValueError):
        statemachine_init.actual_fen_move(impossible_move)

def test_actual_fen_move_from_start(statemachine_init):
    first_move = "h2h3"
    first_result_state = "8/8/8/8/8/7K/krbnNBR1/qrbnNBRQ b - - 1 1"
    statemachine_init.actual_fen_move(first_move)
    assert statemachine_init.actual_game.board.fen() == first_result_state

def test_rollout_idx_move_from_second(statemachine_init):
    first_move = "h2h3"
    statemachine_init.actual_fen_move(first_move)
    second_move = [0, 6, 0]
    second_result_state = "8/8/8/8/8/k6K/1rbnNBR1/qrbnNBRQ w - - 2 2"
    statemachine_init.idx_move(second_move)
    assert statemachine_init.rollout_game.board.fen() == second_result_state

def test_actual_state_is_unaffected_by_rollout_move(statemachine_init):
    first_move = "h2h3"
    first_result_state = "8/8/8/8/8/7K/krbnNBR1/qrbnNBRQ b - - 1 1"
    statemachine_init.actual_fen_move(first_move)
    second_move = [0, 6, 0]
    second_result_state = "8/8/8/8/8/k6K/1rbnNBR1/qrbnNBRQ w - - 2 2"
    statemachine_init.idx_move(second_move)
    assert statemachine_init.actual_game.board.fen() == first_result_state

def test_actual_idx_move_from_second(statemachine_init):
    first_move = "h2h3"
    first_result_state = "8/8/8/8/8/7K/krbnNBR1/qrbnNBRQ b - - 1 1"
    statemachine_init.actual_fen_move(first_move)
    second_move = [0, 6, 0]
    second_result_state = "8/8/8/8/8/k6K/1rbnNBR1/qrbnNBRQ w - - 2 2"
    statemachine_init.actual_idx_move(second_move)
    assert statemachine_init.actual_game.board.fen() == second_result_state

def test_set_to_fen_state(statemachine_stalemate):
    new_state = "8/8/8/8/8/8/R7/5K1k b - - 10 20"
    assert statemachine_stalemate.actual_game.board.fen() == new_state

def test_get_player_color_from_new_state(statemachine_stalemate):
    assert statemachine_stalemate.get_player_color() == BLACK

def test_get_legal_moves_in_no_valid_position(statemachine_stalemate):
    num_of_legal_moves_from_new_state = 0
    moves = statemachine_stalemate.get_legal_moves()
    assert len(moves[0]) == num_of_legal_moves_from_new_state

def test_get_rollout_result_from_no_valid_move(statemachine_stalemate):
    assert statemachine_stalemate.get_result() == 0

def test_get_actual_result_from_no_valid_move(statemachine_stalemate):
    assert statemachine_stalemate.get_actual_result() == 0

def test_win_result_black(statemachine_win):
    assert statemachine_win.get_actual_result() == -1

def test_win_game_game_over(statemachine_win):
    assert statemachine_win.actual_game_over() == True

def test_suspended_rollout_is_also_set(statemachine_suspended):
    suspended_win = statemachine_suspended.actual_game.board.fen()
    assert statemachine_suspended.rollout_game.board.fen() == suspended_win

def test_suspended_result(statemachine_suspended):
    assert statemachine_suspended.get_actual_result() == 0

def test_suspended_has_not_ended(statemachine_suspended):
    assert statemachine_suspended.actual_game_over() == False

def test_black_suspended_is_win(sm_noblacksuspense):
    assert sm_noblacksuspense.get_actual_result() == BLACK

def test_black_suspended_game_over(sm_noblacksuspense):
    assert sm_noblacksuspense.actual_game_over() == True

# pylint: enable=C0116
