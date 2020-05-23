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


def test_end_on_stalemate(stalemate):
    assert stalemate.simulate() == "draw by stale mate"

def test_end_on_repetition(repetition):
    assert repetition.simulate() == "draw by repetition"


