from azts import stateMachine as sm
from azts.config import *

state_machine = sm.StateMachine() 

num_of_legal_moves_from_start = 21

def test_get_player_color_from_start():
    assert state_machine.get_player_color() == WHITE

def test_get_legal_moves_right_number_of_moves():
    moves = state_machine.get_legal_moves()
    assert len(moves[0]) == num_of_legal_moves_from_start

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


