import pytest
from azts import self_match

from azts.config import *

@pytest.fixture
def stalemate():
    stale_mate = "8/8/8/8/8/8/R7/5K1k b - - 10 20" 
    match = self_match.SelfMatch()
    match.set_game_state(stale_mate)
    return match 

@pytest.fixture
def repetition():
    startpos = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"
    match = self_match.SelfMatch()
    match.game.history[startpos] = 5
    return match

@pytest.fixture
def black_win():
    win_state = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    match = self_match.SelfMatch()
    match.set_game_state(win_state)
    return match

@pytest.fixture
def white_win():
    win_state = "7K/8/8/8/8/8/R7/5k2 b - - 10 20"
    match = self_match.SelfMatch()
    match.set_game_state(win_state)
    return match

@pytest.fixture
def suspension():
    suspended_win = "7K/8/k7/8/8/8/7R/8 b - - 10 20"
    match = self_match.SelfMatch()
    match.set_game_state(suspended_win)
    return match

@pytest.fixture
def suspension_draw():
    suspended_draw = "7K/k7/7R/8/8/8/8/1R6 b - - 10 20"
    match = self_match.SelfMatch()
    match.set_game_state(suspended_draw)
    return match

def test_end_on_stalemate(stalemate):
    assert stalemate.simulate() == DRAW_BY_STALE_MATE

def test_end_on_repetition(repetition):
    assert repetition.simulate() == DRAW_BY_REP

def test_end_on_black_win(black_win):
    assert black_win.simulate() == BLACK_WINS

def test_end_on_white_win(white_win):
    assert white_win.simulate() == WHITE_WINS

def test_suspended_white_win(suspension):
    assert suspension.simulate() == WHITE_WINS

def test_suspended_draw(suspension_draw):
    assert suspension_draw.simulate() == DRAW_BY_TWO_WINS
