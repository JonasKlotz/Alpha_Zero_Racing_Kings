# pylint: disable=E0401
'''
player class representing
an ai player with a simple
API.
'''

from Player import config

from azts import azts_tree
from azts import state_machine
from azts import mock_model

from azts.config import RUNS_PER_MOVE, WHITE, \
        EXPLORATION, ROLLOUT_PAYOFFS, HEAT, \
        MODEL

class Player():
    '''
    Player that keeps track of
    game states and generates moves
    with an azts tree and a neural
    network model.
    A player is initialized with a
    player color (-1 for black, 1 for
    white) and the number of rollouts
    to be made in the alpha zero tree
    search for every move
    '''
    def __init__(self, \
            color=WHITE, \
            model=MODEL, \
            runs_per_move=RUNS_PER_MOVE, \
            exploration=EXPLORATION, \
            rollout_payoffs=ROLLOUT_PAYOFFS, \
            heat=HEAT, \
            **kwargs):

        # player is actually not keeping any state,
        # so no need to store statemachine or model
        # in self
        statemachine = state_machine.StateMachine()
        self.tree = azts_tree.AztsTree(statemachine=statemachine, \
                              model=model, \
                              color=color, \
                              runs_per_move=runs_per_move, \
                              exploration=exploration, \
                              payoffs=rollout_payoffs, \
                              heat=heat)

    def set_color(self, color):
        '''
        sets color =)
        '''
        self.tree.set_color(color)

    def make_move(self):
        '''
        :return str: move in uci notation
        '''
        return self.tree.make_move()

    def receive_move(self, move):
        '''
        update inner state according to other
        players move
        :param str move: move in uci notation
        '''
        self.tree.receive_move(move)

    def game_over(self):
        '''
        check if player thinks that game
        is over
        :return boolean: True if game is over
        '''
        return self.tree.game_over()

    def game_state(self):
        '''
        check state of inner game according
        to enum types as defined in config
        :return int: running, white wins,
        black wins, draw, draw by stale mate,
        draw by repetition, draw by two wins
        '''
        return self.tree.game_state()

    def set_game_state(self, fen_position):
        '''
        set inner state to a state provided by
        a fen string
        :param str fen_position: state to set
        the player to
        '''
        self.tree.set_to_fen_state(fen_position)

    def get_stats(self):
        '''
        get statistics about azts tree search
        parameters
        :return dict: dictionary containing
        information about tree shape (max
        depth, num of end states etc.) and
        move distribution (probability of
        best move etc.)
        '''
        return self.tree.get_move_statistics()

    def dump_data(self):
        '''
        poll for internal data
        :return list: with two entries; first
        is current game position in tensor notation
        as np.array; second is current policy
        tensor as np.array
        '''
        return [self.tree.get_position(), \
                self.tree.get_policy_tensor(), \
                None]


if __name__ == "__main__":
    model = mock_model.MockModel()
    configuration = config.Config("Player/default_config.yaml")
    print(configuration.player.as_dictionary())
    player = Player(model=model, \
            **(configuration.player.as_dictionary()))
    print(f"First move of white player is {player.make_move()}.")
    print(player.get_stats())
# pylint: enable=E0401
