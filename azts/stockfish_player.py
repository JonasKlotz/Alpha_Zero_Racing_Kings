import sys
import os

from Interface.TensorNotation import fen_to_tensor
from Player import config
import chess
from azts.config import ROOTDIR
import platform
from azts import state_machine

from azts.config import RUNS_PER_MOVE, WHITE, BLACK
from azts.player import Player


class StockfishPlayer(Player):
    '''
    class that represents a human player.
    calls to make_move() actually trigger
    communication with the player over
    a command line interface (cli).
    '''
    ENGINE = {"Linux": "stockfish-x86_64", \
              "Darwin": "stockfish-osx-x86_64", \
              "Windows": "stockfish-windows-x86_64.exe"}
    PATH_TO_ENGINE = os.path.join(ROOTDIR, "Interpreter")
    PATH_TO_ENGINE = os.path.join(PATH_TO_ENGINE, "Engine")
    PATH_TO_ENGINE = os.path.join(PATH_TO_ENGINE, ENGINE[platform.system()])
    ENGINE = {"Linux": "stockfish-x86_64", \
              "Darwin": "stockfish-osx-x86_64", \
              "Windows": "stockfish-windows-x86_64.exe"}
    limit = 0.1
    pol = None

    def __init__(self, name="THEMIGHTYFISH", color=WHITE, **kwargs):
        '''
        CLIPlayer is indeed keeping state,
        because there is no azts_tree
        involved to keep state
        '''
        # super().__init__(name, color, **kwargs) # necessary??
        self.name = name
        self.color = color
        self.statemachine = state_machine.StateMachine()

    def set_color(self, color):
        self.color = color

    def set_runs_per_move(self, runs_per_move):
        pass

    def reset(self):
        '''
        reset all stateful things
        which is only the statemachine
        in command line players
        '''
        self.statemachine = state_machine.StateMachine()

    def make_move(self):
        '''
        poll the player for a move
        '''

        self.pol = self.statemachine.actual_game.get_policy(self.PATH_TO_ENGINE, self.limit)
        self.pol = self.statemachine.actual_game.normalize_policy_zero_one(self.pol)
        move = self.pol[0][0]
        self.receive_move(move)
        return move

    def receive_move(self, move):
        '''
        update own state machine and
        print feedback to player
        '''

        # if self.color is not self.statemachine.get_player_color():
        #     print(f"> Other player played {move}")

        self.statemachine.actual_fen_move(move)

    # TODO: implement other getters and setters

    def game_over(self):
        return self.statemachine.actual_game_over()

    def get_stats(self):
        return None

    def dump_data(self):
        fen = self.statemachine.get_actual_fen_position()
        fen_tensor = fen_to_tensor(fen)

        pol_tensor = self.statemachine.actual_game.policy_to_tensor(self.pol)
        return [fen_tensor, pol_tensor, None]
