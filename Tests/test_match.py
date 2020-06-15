# pylint: disable=E0401
# pylint: disable=E0602
# pylint: disable=C0111
# pylint: disable=W0621
import pytest

from Azts import mock_model
from Azts import player
from Azts import utility 
from Azts.config import DRAW_BY_STALE_MATE, \
        DRAW_BY_REP, DRAW_BY_TWO_WINS, \
        BLACK_WINS, WHITE_WINS
from Matches import match

@pytest.fixture
def new_match():
    model = mock_model.MockModel()
    conf = utility.load_player_conf("Player/default_config")
    player_one = player.Player(model=model, \
            rollouts_per_move=10, \
            **(conf.player.as_dictionary()))
    player_two = player.Player(model=model, \
            rollouts_per_move=10, \
            **(conf.player.as_dictionary()))
    new_match = match.Match(player_one, player_two)
    return new_match

@pytest.fixture
def stalemate(new_match):
    stale_mate = "8/8/8/8/8/8/R7/5K1k b - - 10 20"
    new_match.set_game_state(stale_mate)
    return new_match

@pytest.fixture
def repetition(new_match):
    startpos = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"
    new_match.game.history[startpos.split(" ")[0]] = 5
    return new_match

@pytest.fixture
def black_win(new_match):
    win_state = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    new_match.set_game_state(win_state)
    return new_match

@pytest.fixture
def white_win(new_match):
    win_state = "7K/8/8/8/8/8/R7/5k2 b - - 10 20"
    new_match.set_game_state(win_state)
    return new_match

@pytest.fixture
def suspension(new_match):
    suspended_win = "7K/8/k7/8/8/8/7R/8 b - - 10 20"
    new_match.set_game_state(suspended_win)
    return new_match

@pytest.fixture
def suspension_draw(new_match):
    suspended_draw = "7K/k7/7R/8/8/8/8/1R6 b - - 10 20"
    new_match.set_game_state(suspended_draw)
    return new_match

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
