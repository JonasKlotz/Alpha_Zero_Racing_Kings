from azts import stateMachine as sm
from azts.config import *

state_machine = sm.StateMachine() 

num_of_legal_moves_from_start = 21

def test_get_player_color_from_start():
    assert state_machine.get_player_color() == WHITE

def test_get_legal_moves_right_number_of_moves():
    moves = state_machine.get_legal_moves()
    assert len(moves[0]) == num_of_legal_moves_from_start

def test_get_rollout_result_from_start():
    assert state_machine.get_rollout_result() == 0

def test_get_actual_result_from_start():
    assert state_machine.get_actual_result() == 0

first_move = "h2h3"
first_result_state = "8/8/8/8/8/7K/krbnNBR1/qrbnNBRQ b - - 1 1"

def test_actual_fen_move_from_start():
    state_machine.actual_fen_move(first_move)
    assert state_machine.actual_game.board.fen() == first_result_state

second_move = [0, 6, 0]
second_result_state = "8/8/8/8/8/k6K/1rbnNBR1/qrbnNBRQ w - - 2 2"

def test_rollout_idx_move_from_second():
    state_machine.rollout_idx_move(second_move)
    assert state_machine.rollout_game.board.fen() == second_result_state

def test_actual_state_is_unaffected_by_rollout_move(): 
    assert state_machine.actual_game.board.fen() == first_result_state

def test_actual_idx_move_from_second():
    state_machine.actual_idx_move(second_move)
    assert state_machine.actual_game.board.fen() == second_result_state

new_state = "8/8/8/8/8/8/R7/5K1k b - - 10 20" 

def test_set_to_fen_state():
    state_machine.set_to_fen_state(new_state)
    assert state_machine.actual_game.board.fen() == new_state

def test_get_player_color_from_new_state():
    assert state_machine.get_player_color() == BLACK 

num_of_legal_moves_from_new_state = 0

def test_get_legal_moves_in_no_valid_position():
    moves = state_machine.get_legal_moves()
    assert len(moves[0]) == num_of_legal_moves_from_new_state 

def test_get_rollout_result_from_no_valid_move():
    assert state_machine.get_rollout_result() == 0

def test_get_actual_result_from_no_valid_move():
    assert state_machine.get_actual_result() == 0



