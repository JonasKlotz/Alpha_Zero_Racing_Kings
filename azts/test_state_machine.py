# pylint: disable=C0116
import pytest

from azts import state_machine 
from azts.config import WHITE, BLACK 

statemachine = state_machine.StateMachine()

def test_two_games_are_independent():
    assert statemachine.actual_game is not \
        statemachine.rollout_game

NUM_OF_LEGAL_MOVES_FROM_START = 21

def test_correct_tensor_dimensions_for_position():
    assert statemachine.get_position().shape == (8, 8, 11)

def test_get_player_color_from_start():
    assert statemachine.get_player_color() == WHITE

def test_start_game_has_not_ended():
    assert statemachine.actual_has_ended() == False

def test_get_legal_moves_right_number_of_moves():
    moves = statemachine.get_legal_moves()
    assert len(moves[0]) == NUM_OF_LEGAL_MOVES_FROM_START

def test_get_rollout_result_from_start():
    assert statemachine.get_result() == 0

def test_get_actual_result_from_start():
    assert statemachine.get_actual_result() == 0

IMPOSSIBLE_MOVE = "h1h8"

def test_exception_on_impossible_move():
    with pytest.raises(ValueError):
        statemachine.actual_fen_move(IMPOSSIBLE_MOVE)

FIRST_MOVE = "h2h3"
FIRST_RESULT_STATE = "8/8/8/8/8/7K/krbnNBR1/qrbnNBRQ b - - 1 1"

def test_actual_fen_move_from_start():
    statemachine.actual_fen_move(FIRST_MOVE)
    assert statemachine.actual_game.board.fen() == FIRST_RESULT_STATE

SECOND_MOVE = [0, 6, 0]
SECOND_RESULT_STATE = "8/8/8/8/8/k6K/1rbnNBR1/qrbnNBRQ w - - 2 2"

def test_rollout_idx_move_from_second():
    statemachine.idx_move(SECOND_MOVE)
    assert statemachine.rollout_game.board.fen() == SECOND_RESULT_STATE

def test_actual_state_is_unaffected_by_rollout_move():
    assert statemachine.actual_game.board.fen() == FIRST_RESULT_STATE

def test_actual_idx_move_from_second():
    statemachine.actual_idx_move(SECOND_MOVE)
    assert statemachine.actual_game.board.fen() == SECOND_RESULT_STATE

NEW_STATE = "8/8/8/8/8/8/R7/5K1k b - - 10 20"

def test_set_to_fen_state():
    statemachine.set_to_fen_state(NEW_STATE)
    assert statemachine.actual_game.board.fen() == NEW_STATE

def test_get_player_color_from_new_state():
    assert statemachine.get_player_color() == BLACK

NUM_OF_LEGAL_MOVES_FROM_NEW_STATE = 0

def test_get_legal_moves_in_no_valid_position():
    moves = statemachine.get_legal_moves()
    assert len(moves[0]) == NUM_OF_LEGAL_MOVES_FROM_NEW_STATE

def test_get_rollout_result_from_no_valid_move():
    assert statemachine.get_result() == 0

def test_get_actual_result_from_no_valid_move():
    assert statemachine.get_actual_result() == 0

WIN_STATE = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
win_statemachine = state_machine.StateMachine()
win_statemachine.set_to_fen_state(WIN_STATE)

def test_win_result_black():
    assert win_statemachine.get_actual_result() == -1

def test_win_game_has_ended():
    assert win_statemachine.actual_has_ended() == True

SUSPENDED_WIN = "7K/k7/8/8/8/8/7R/8 b - - 10 20"
suspended_statemachine = state_machine.StateMachine()
suspended_statemachine.set_to_fen_state(SUSPENDED_WIN)

def test_suspended_rollout_is_also_set():
    assert suspended_statemachine.rollout_game.board.fen() == SUSPENDED_WIN

def test_suspended_result():
    assert suspended_statemachine.get_actual_result() == 0

def test_suspended_has_not_ended():
    assert suspended_statemachine.actual_has_ended() == False

NO_BLACK_SUSPENSE = "7k/K7/8/8/8/8/R7/8 w - - 10 20"
no_suspense_statemachine = state_machine.StateMachine()
no_suspense_statemachine.set_to_fen_state(NO_BLACK_SUSPENSE)

def test_black_suspended_is_win():
    assert no_suspense_statemachine.get_actual_result() == BLACK

def test_black_suspended_has_ended():
    assert no_suspense_statemachine.actual_has_ended() == True

# pylint: enable=C0116
