# pylint: disable=E0401
'''
player class representing
an ai player with a simple
API.
'''
import sys

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
            name="UNNAMED PLAYER", \
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
        self.name = name
        self.tree = azts_tree.AztsTree(model=model, \
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

    def reset(self):
        '''
        resets all stateful things
        '''
        self.tree.reset()

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


class CLIPlayer(Player):

    def __init__(self, \
            name="UNNAMED PLAYER", \
            color=WHITE):

        # player is actually not keeping any state,
        # so no need to store statemachine or model
        # in self
        self.name = name
        self.color = color
        self.statemachine = state_machine.StateMachine()

    def set_color(self, color):
        self.color = color

    def reset(self):
        '''
        reset all stateful things
        which is only the statemachine
        in command line players
        '''
        self.statemachine = state_machine.StateMachine()

    def make_move(self):
        move = self._parse_user_input()
        self.receive_move(move)
        return move

    def _parse_user_input(self):
        position = self.statemachine.get_actual_fen_position()
        print(f"> current state is {position}.")
        print("> select move in UCI or type\n" \
                + "> \"help\" for help\n" \
                + "> \"list\" to list all possible moves\n" \
                + "> \"exit\" to exit")

        user_input = "unknown"

        legal_moves = self.statemachine.rollout_game.get_moves_observation()
        choices = legal_moves + ["help", "list", "exit"]


        while user_input not in choices:
            user_input = input("> ")

        if user_input in legal_moves:
            return user_input

        consequences = {"help": lambda x: print("Try to " \
                + "put your king on the last rank. Prevent " \
                + "your opponent from doing the same.\n"), \
                "exit": lambda x: sys.exit(), \
                "list": lambda x: [print(i) for i in x]}

        consequences[user_input](legal_moves)
        
        return self._parse_user_input()


    def receive_move(self, move):
        print(f"Other player played {move}")
        self.statemachine.actual_fen_move(move)

    #TODO: implement other getters and setters

    def game_over(self):
        return self.statemachine.actual_game_over()

    def get_stats(self):
        return None

    def dump_data(self):
        return [None, None, None]





if __name__ == "__main__":
    model = mock_model.MockModel()
    configuration = config.Config("Player/default_config.yaml")
    player = Player(model=model, \
            **(configuration.player.as_dictionary()))
    print(f"First move of white player is {player.make_move()}.")
    print(player.get_stats())
# pylint: enable=E0401
