# pylint: disable=E0401
# pylint: disable=E0602
# pylint: disable=C0111
# pylint: disable=W0621
import pytest
from azts import self_match
from azts import mock_model
from azts import player

from azts.config import DRAW_BY_STALE_MATE, \
        DRAW_BY_REP, DRAW_BY_TWO_WINS, \
        BLACK_WINS, WHITE_WINS

@pytest.fixture
def stalemate():
    model = mock_model.MockModel()
    player_one = player.Player(model=model)
    player_two = player.Player(model=model) 
    stale_mate = "8/8/8/8/8/8/R7/5K1k b - - 10 20"
    match = self_match.SelfMatch(player_one, player_two)
    match.set_game_state(stale_mate)
    return match

@pytest.fixture
def repetition():
    model = mock_model.MockModel()
    player_one = player.Player(model=model)
    player_two = player.Player(model=model) 
    startpos = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"
    match = self_match.SelfMatch(player_one, player_two)
    match.game.history[startpos.split(" ")[0]] = 5
    return match

@pytest.fixture
def black_win():
    model = mock_model.MockModel()
    player_one = player.Player(model=model)
    player_two = player.Player(model=model) 
    win_state = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    match = self_match.SelfMatch(player_one, player_two)
    match.set_game_state(win_state)
    return match

@pytest.fixture
def white_win():
    model = mock_model.MockModel()
    player_one = player.Player(model=model)
    player_two = player.Player(model=model) 
    win_state = "7K/8/8/8/8/8/R7/5k2 b - - 10 20"
    match = self_match.SelfMatch(player_one, player_two)
    match.set_game_state(win_state)
    return match

@pytest.fixture
def suspension():
    model = mock_model.MockModel()
    player_one = player.Player(model=model)
    player_two = player.Player(model=model) 
    suspended_win = "7K/8/k7/8/8/8/7R/8 b - - 10 20"
    match = self_match.SelfMatch(player_one, player_two)
    match.set_game_state(suspended_win)
    return match

@pytest.fixture
def suspension_draw():
    model = mock_model.MockModel()
    player_one = player.Player(model=model)
    player_two = player.Player(model=model) 
    suspended_draw = "7K/k7/7R/8/8/8/8/1R6 b - - 10 20"
    match = self_match.SelfMatch(player_one, player_two)
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
# pylint: enable=E0401
# pylint: enable=E0602
# pylint: enable=C0111
# pylint: enable=W0621
