import pytest
from azts import player

from azts.config import WHITE

@pytest.fixture
def player_white():
    p1 = player.Player(WHITE, 10)
    return p1

def test_player_move(player_white):
    legal_moves = \
        player_white.tree.statemachine.actual_game.get_moves_observation()
    move = player_white.make_move()
    assert move in legal_moves

