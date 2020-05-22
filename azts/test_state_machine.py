from azts import state_machine as sm
from azts.config import WHITE, BLACK

# pylint: disable=C0116

state_machine = sm.StateMachine()

NUM_OF_LEGAL_MOVES_FROM_START = 21

def test_get_player_color_from_start():
    assert state_machine.get_player_color() == WHITE

def test_get_legal_moves_right_number_of_moves():
    moves = state_machine.get_legal_moves()
    assert len(moves[0]) == NUM_OF_LEGAL_MOVES_FROM_START

def test_get_rollout_result_from_start():
    assert state_machine.get_rollout_result() == 0

def test_get_actual_result_from_start():
    assert state_machine.get_actual_result() == 0

FIRST_MOVE = "h2h3"
FIRST_RESULT_STATE = "8/8/8/8/8/7K/krbnNBR1/qrbnNBRQ b - - 1 1"

def test_actual_fen_move_from_start():
    state_machine.actual_fen_move(FIRST_MOVE)
    assert state_machine.actual_game.board.fen() == FIRST_RESULT_STATE

SECOND_MOVE = [0, 6, 0]
SECOND_RESULT_STATE = "8/8/8/8/8/k6K/1rbnNBR1/qrbnNBRQ w - - 2 2"

def test_rollout_idx_move_from_second():
    state_machine.rollout_idx_move(SECOND_MOVE)
    assert state_machine.rollout_game.board.fen() == SECOND_RESULT_STATE

def test_actual_state_is_unaffected_by_rollout_move():
    assert state_machine.actual_game.board.fen() == FIRST_RESULT_STATE

def test_actual_idx_move_from_second():
    state_machine.actual_idx_move(SECOND_MOVE)
    assert state_machine.actual_game.board.fen() == SECOND_RESULT_STATE

NEW_STATE = "8/8/8/8/8/8/R7/5K1k b - - 10 20"

def test_set_to_fen_state():
    state_machine.set_to_fen_state(NEW_STATE)
    assert state_machine.actual_game.board.fen() == NEW_STATE

def test_get_player_color_from_new_state():
    assert state_machine.get_player_color() == BLACK

NUM_OF_LEGAL_MOVES_FROM_NEW_STATE = 0

def test_get_legal_moves_in_no_valid_position():
    moves = state_machine.get_legal_moves()
    assert len(moves[0]) == NUM_OF_LEGAL_MOVES_FROM_NEW_STATE

def test_get_rollout_result_from_no_valid_move():
    assert state_machine.get_rollout_result() == 0

def test_get_actual_result_from_no_valid_move():
    assert state_machine.get_actual_result() == 0
# pylint: enable=C0116
