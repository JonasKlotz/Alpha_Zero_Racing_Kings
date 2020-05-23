import pytest
from azts import player

from azts.config import WHITE, BLACK

@pytest.fixture
def player_white():
    player_white = player.Player(WHITE, 10)
    return player_white

@pytest.fixture
def other_players_turn():
    player_black = player.Player(BLACK, 10)
    return player_black

@pytest.fixture
def stale_mate():
    player_black = player.Player(BLACK, 10) 
    new_state = "8/8/8/8/8/8/R7/5K1k b - - 10 20" 
    player_black.set_game_state(new_state)
    return player_black

@pytest.fixture
def win_position():
    player_black = player.Player(BLACK, 10)
    win_position = "7k/8/8/8/8/8/R7/5K2 w - - 10 20"
    player_black.set_game_state(win_position)
    return player_black

def test_player_move(player_white):
    legal_moves = \
        player_white.tree.statemachine.actual_game.get_moves_observation()
    move = player_white.make_move()
    assert move in legal_moves

def test_game_not_over(player_white):
    assert player_white.game_over() == False

def test_game_running(player_white):
    assert player_white.game_state() == "running"

def test_dont_receive_move(player_white):
    move = "a2a3"
    with pytest.raises(Exception):
        player_white.receive_move(move)

def test_dump_data_len_is_three(player_white):
    assert len(player_white.dump_data()) == 3

def test_dump_data_type_is_list(player_white):
    assert type(player_white.dump_data()) is list

def test_dont_accept_illegal_move(other_players_turn):
    illegal_move = "a1a8"
    with pytest.raises(Exception):
        other_players_turn.receive_move(illegal_move)

def test_accept_legal_move(other_players_turn):
    move = "h2h3"
    other_players_turn.receive_move(move)
    assert other_players_turn.game_state() == "running"

def test_stale_mate_game_over(stale_mate):
    assert stale_mate.game_over()

def test_stale_mate_game_state(stale_mate):
    assert stale_mate.game_state() == "draw by stale mate"

def test_no_move_in_stale_mate(stale_mate):
    with pytest.raises(Exception):
        stale_mate.make_move()

def test_dont_receive_move_in_stale_mate(stale_mate):
    move = "a2a8"
    with pytest.raises(Exception):
        stale_mate.receive_move()

def test_dont_receive_illegal_move_in_stale_mate(stale_mate):
    move = "a2h8"
    with pytest.raises(Exception):
        stale_mate.receive_move()

def test_game_over_in_win_position(win_position):
    assert win_position.game_over()

def test_game_state_in_win_position(win_position):
    assert win_position.game_state() == "black won"

def test_won_no_move(win_position):
    with pytest.raises(Exception):
        win_position.make_move()

def test_won_dont_accept_move(win_position):
    move = "a2a7"
    with pytest.raises(Exception):
        win_position.receive_move(move)

def test_won_dont_accept_illegal_move(win_position):
    move = "a2c7"
    with pytest.raises(Exception):
        win_position.receive_move(move)