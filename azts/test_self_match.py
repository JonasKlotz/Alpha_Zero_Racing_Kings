import pytest
from azts import self_match

from azts.config import WHITE, BLACK

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
    assert stalemate.simulate() == "draw by stale mate"

def test_end_on_repetition(repetition):
    assert repetition.simulate() == "draw by repetition"

def test_end_on_black_win(black_win):
    assert black_win.simulate() == "black won"

def test_end_on_white_win(white_win):
    assert white_win.simulate() == "white won"

def test_suspended_white_win(suspension):
    assert suspension.simulate() == "white won"

def test_suspended_draw(suspension_draw):
    assert suspension_draw.simulate() == "draw by two finishes in one turn"
